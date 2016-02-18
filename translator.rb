require_relative "types/primitive_type"
require_relative "types/ruby_extension"
require_relative "scope"
require_relative "parsing"
require "set"
require "pp"
require "tempfile"
require "ffi"
require "parser/current"

class Translator
    class TranslationResult
        def initialize(c_source, type)
            @c_source = c_source
            @type = type
        end
        
        def c_source
            @c_source
        end
        
        def type
            @type
        end
    end
    
    class CommandProxy
        attr_accessor :c_source
        attr_accessor :result_type
        attr_accessor :array_size
        
        def initialize
            @env_size = 0
            @env = {}
            @ffi_wrapper = nil
            @expected_input_types = []
        end
        
        def build
            file = Tempfile.new(["ikra_kernel", ".cu"])
            file.write(c_source)
            file.close
            
            compile_status = %x(nvcc -o #{file.path}.dll --shared #{file.path})
            
            @ffi_wrapper = Module.new
            @ffi_wrapper.extend(FFI::Library)
            @ffi_wrapper.ffi_lib(file.path + ".dll")
            @ffi_wrapper.attach_function(:launch_kernel, [:pointer] * (@expected_input_types.size + 1), :pointer)
        end
        
        def add_input_type(type)
            @expected_input_types.push(type)
        end
        
        def add_env_var(offset, size, accessor)
            @env[offset] = accessor
            @env_size += size
        end
        
        def execute(*input)
            if input.size != @expected_input_types.size
                raise "Expected #{@expected_input_types.size} parameters (#{input.size} given)"
            end
            
            env = FFI::MemoryPointer.new(@env_size)
            @env.each do |offset, accessor|
                value = accessor.call
                if value.class == Float
                    env.put_float(offset, value)
                elsif value.class == Fixnum
                    env.put_int(offset, value)
                else
                    raise "Cannot pass object of type #{value.class} via FFI"
                end
            end
            
            all_input = [env]
            (0..input.size - 1).each do |index|
                arg = FFI::MemoryPointer.new(input[index].size * @expected_input_types[index].c_size)
                if @expected_input_types[index] == PrimitiveType::Int
                    arg.put_array_of_int(0, input[index])
                elsif @expected_input_types[index] == PrimitiveType::Float
                    arg.put_array_of_float(0, input[index])
                else
                    raise "Cannot handle array of #{@result_type} via FFI"
                end
                all_input.push(arg)
            end
            
            result = @ffi_wrapper.launch_kernel(*all_input)
            
            all_input.each do |pointer|
                pointer.free
            end
            
            if @result_type == PrimitiveType::Int
                result.read_array_of_int(@array_size)
            elsif @result_type == PrimitiveType::Float
                result.read_array_of_float(@array_size)
            else
                raise "Cannot retrieve array of #{@result_type} via FFI"
            end
        end
    end
    
    class InputVariable
        def initialize(name, type, is_index = false)
            @name = name
            @type = type
            @is_index = is_index
        end
        
        def name
            @name
        end
        
        def type
            @type
        end
        
        def is_index?
            @is_index
        end
    end
    
    def self.grid_block_size(size)
        [[[size / 250, 1].max, 1, 1], [(size >= 250 ? 250 : size), 1, 1]]
    end
    
    def self.translate_block(block, size, input_vars = [])
        instance = self.new
        instance.symbol_table.push_frame
        
        local_variables = []
        
        # what if variables are passed to the kernel and changed in-place?
        # we can implement locking or reduce-style merge for primitive arithmetic operators
        block.binding.local_variables.each do |var|
            instance.symbol_table.ensure_defined(var, block.binding.local_variable_get(var).class.to_ikra_type)
            local_variables.push(var)
        end
        
        input_vars.each do |var|
            instance.symbol_table.define_shadowed(var.name, var.type)
            local_variables.push(var.name)
        end
        
        ast = Parsing.parse(block, local_variables)
        dimensions = grid_block_size(size)
        command_proxy = instance.translate_function(ast, size, block, input_vars, dimensions[0], dimensions[1])
        
        instance.symbol_table.pop_frame
        
        command_proxy.build
        command_proxy
    end
    
    def initialize
        @symbol_table = Scope.new
    end
    
    def symbol_table
        @symbol_table
    end
    
    EnvironmentVariable = "_env_"
    ResultVariable = "_result_"
    TempResultVariable = "_temp_result_"
    
    # Translates an AST node to C++/CUDA code. Generates the kernel and a kernel launcher.
    #
    # * *Arguments*
    #   - +node+: AST node
    #   - +size+: Size of array (number of threads)
    #   - +block+: Original block (for extracting environment and parameters)
    #   - +input_vars+: Block parameters and accessor blocks
    #   - +dim3_grid+: Dimension of grid
    #   - +dim3_block+: Dimension of block
    #
    def translate_function(node, size, block, input_vars = [], dim3_grid = [1, 1, 1], dim3_block = [size, 1, 1])
        result = nil
        mem_offsets = {}
        lexical_size = 0
        result_type = nil
        launcher_params = ["void *host_parameters"]
        kernel_params = ["void *#{EnvironmentVariable}", "void *#{ResultVariable}"]
        kernel_args = ["device_parameters", "device_result"]
        device_input_decl = ""
        command_proxy = CommandProxy.new
        
        @symbol_table.new_frame do
            result = translate_multi_begin_or_statement(node, true)
            result_type = @symbol_table.get_type(:"#")
            command_proxy.result_type = result_type
            
            result = variable_definitions + result.c_source
            input_var_names = input_vars.map do |var|
                var.name
            end
            
            mem_offset = 0
            # lexical variables
            (@symbol_table.read_and_written_variables(-2) - input_var_names).each do |var|
                type = @symbol_table.get_type(var)
                assignment = "#{type.to_c_type} #{var.to_s} = * (#{type.to_c_type} *) (((char *) #{EnvironmentVariable}) + #{mem_offset.to_s})"
                result = assignment + ";\n" + result
                
                env_accessor = Proc.new do
                    block.binding.local_variable_get(var)
                end
                command_proxy.add_env_var(mem_offset, type.c_size, env_accessor)
                
                mem_offsets[var] = mem_offset
                mem_offset += type.c_size
                lexical_size += type.c_size
            end
            
            # declare temp result variable
            result = "#{result_type.to_c_type} #{TempResultVariable};\n" + result
            
            # array input variables
            input_vars.each do |var|
                if var.is_index?
                    value = "threadIdx.x + blockIdx.x * blockDim.x"
                else
                    value = "_input_#{var.name}[threadIdx.x + blockIdx.x * blockDim.x]"
                    command_proxy.add_input_type(var.type)
                    
                    launcher_params.push("#{var.type.to_c_type} *host_input_#{var.name}")
                    kernel_params.push("#{var.type.to_c_type} *_input_#{var.name}")
                    kernel_args.push("device_input_#{var.name}")
                    device_input_decl += """#{var.type.to_c_type} *device_input_#{var.name};
    cudaMalloc(&device_input_#{var.name}, #{var.type.c_size} * #{size});
    cudaMemcpy(device_input_#{var.name}, host_input_#{var.name}, #{var.type.c_size} * #{size}, cudaMemcpyHostToDevice);
"""
                end
                
                result = "#{var.type.to_c_type} #{var.name} = #{value};\n" + result
            end
        end
        
        input_vars = 
        # generate kernel launcher
        launcher = """#include <stdio.h>
        
extern \"C\" __declspec(dllexport) #{result_type.to_c_type} *launch_kernel(#{launcher_params.join(", ")})
{
    printf(\"kernel launched\\n\");
    void *device_parameters;
    cudaMalloc(&device_parameters, #{lexical_size});
    cudaMemcpy(device_parameters, host_parameters, #{lexical_size}, cudaMemcpyHostToDevice);
    
    #{result_type.to_c_type} *host_result = (#{result_type.to_c_type} *) malloc(#{result_type.c_size} * #{size});
    #{result_type.to_c_type} *device_result;
    cudaMalloc(&device_result, #{result_type.c_size} * #{size});
    
    #{device_input_decl}
    
    dim3 dim_grid(#{dim3_grid[0]}, #{dim3_grid[1]}, #{dim3_grid[2]});
    dim3 dim_block(#{dim3_block[0]}, #{dim3_block[1]}, #{dim3_block[2]});
    
    kernel<<<dim_grid, dim_block>>>(#{kernel_args.join(", ")});
    
    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, #{result_type.c_size} * #{size}, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_parameters);
    
    return host_result;
}"""

        # TODO: check: threadIdx.x + blockIdx.x * blockDim.x < #{size}
        assign_result = "((#{result_type.to_c_type} *) #{ResultVariable})[threadIdx.x + blockIdx.x * blockDim.x] = #{TempResultVariable};\n"
        kernel_source = "__global__ void kernel(#{kernel_params.join(", ")})\n" + wrap_in_c_block(result + "\n" + assign_result)
        c_source = kernel_source + "\n\n" + launcher
        puts c_source
        
        command_proxy.c_source = c_source
        command_proxy.array_size = size
        command_proxy
    end
    
    def variable_definitions
        result_string = @symbol_table.last.reject do |var|
            # Reject special variable used for returns
            var == :"#"
        end.map do |name, var|
            var.type.to_c_type + " " + name.to_s
        end.join(";\n")
        
        if result_string.size > 0
            result_string += ";\n"
        end
        
        result_string
    end
    
    ExpressionStatements = [:int, :float, :bool, :lvar, :lvasgn, :send]
    
    def translate_ast(node, should_return = false)
        send("translate_#{node.type.to_s}".to_sym, node, should_return)
    end
    
    def translate_int(node, should_return = false)
        TranslationResult.new(node.children[0], PrimitiveType::Int)
    end
    
    def translate_float(node, should_return = false)
        TranslationResult.new(node.children[0], PrimitiveType::Float)
    end
    
    def translate_bool(node, should_return = false)
        TranslationResult.new(node.children[0], PrimitiveType::Bool)
    end
    
    def translate_lvar(node, should_return = false)
        variable_name = node.children[0]
        @symbol_table.read!(variable_name)
        TranslationResult.new(variable_name.to_s, @symbol_table.get_type(variable_name))
    end
    
    def translate_lvasgn(node, should_return = false)
        variable_name = node.children[0]
        value = translate_ast(node.children[1])
        
        @symbol_table.ensure_defined(variable_name, value.type)
        @symbol_table.written!(variable_name)
        TranslationResult.new("#{variable_name} = #{value.c_source}", value.type)
    end
    
    def translate_if(node, should_return = false)
        # TODO: handle inline IF expressions
        
        condition = translate_ast(node.children[0])
        if condition.type != PrimitiveType::Bool
            raise "Expected boolean expression for IF condition"
        end
        
        result_string = "if (#{condition.c_source})\n" + wrap_in_c_block(translate_multi_begin_or_statement(node.children[1], should_return).c_source)
        
        if (node.children[2] != nil)
            result_string += "else\n" + wrap_in_c_block(translate_multi_begin_or_statement(node.children[2], should_return).c_source)
        end
        
        TranslationResult.new(result_string, PrimitiveType::Void)
    end
    
    def translate_for(node, should_return = false)
        if node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :irange
            # TODO: this is the only kind of for loop we can handle right now
            variable_name = node.children[0].children[0]
            range = extract_begin_single_statement(node.children[1])
            range_from = translate_ast(range.children[0])
            range_to = translate_ast(range.children[1])
            
            if range_from.type != PrimitiveType::Int or range_to.type != PrimitiveType::Int
                raise "Expected range with only integers but found #{range_from.type} and #{range_to.type}"
            end
            
            @symbol_table.ensure_defined(variable_name, PrimitiveType::Int)
            
            result_string = "for (#{variable_name} = (#{range_from.c_source}); #{variable_name} <= (#{range_to.c_source}); #{variable_name}++)\n" + 
                wrap_in_c_block(translate_multi_begin_or_statement(node.children[2]).c_source)
               
            TranslationResult.new(result_string, PrimitiveType::Void)
        end
        
    end
    
    def translate_break(node, should_return = false)
        TranslationResult.new("break", PrimitiveType::Void)
    end
    
    def extract_begin_single_statement(node, should_return = false)
        next_node = node
        while next_node.type == :begin
            if next_node.children.size != 1
                raise "Begin node contains more than one statement"
            end
            
            next_node = next_node.children[0]
        end
        
        next_node
    end
    
    def translate_send(node, should_return = false)
        arith_operators = [:+, :-, :*, :/, :%]
        compare_operators = [:<, :<=, :>, :>=]
        equality_operators = [:==, :!=]
        logic_operators = [:&, :'&&', :|, :'||', :^]
        primitive_operators = arith_operators + compare_operators + equality_operators + logic_operators
        selector = node.children[1]
        receiver = translate_ast(node.children[0])
        
        if primitive_operators.include?(selector)
            operand = translate_ast(node.children[2])
            arg_types = [receiver.type, operand.type]
            result_type = nil
            
            if arith_operators.include?(selector)
                type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Int,
                    [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Float,
                    [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Float}
                
                if type_mapping.has_key?(arg_types)
                    result_type = type_mapping[arg_types]
                elsif type_mapping.has_key?(arg_types.reverse)
                    result_type = type_mapping[arg_types.reverse]
                else
                    raise "Types #{receiver.type} and #{operand.type} not applicable for primitive operator #{selector.to_s}"
                end
            elsif compare_operators.include?(selector)
                type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Bool,
                    [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Bool,
                    [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Bool}
                
                if type_mapping.has_key?(arg_types)
                    result_type = type_mapping[arg_types]
                elsif type_mapping.has_key?(arg_types.reverse)
                    result_type = type_mapping[arg_types.reverse]
                else
                    raise "Types #{receiver.type} and #{operand.type} not applicable for primitive operator #{selector.to_s}"
                end
            elsif equality_operators.include?(selector)
                type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => PrimitiveType::Bool,
                    [PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Bool,
                    [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Bool,
                    [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Bool}
                    
                if type_mapping.has_key?(arg_types)
                    result_type = type_mapping[arg_types]
                elsif type_mapping.has_key?(arg_types.reverse)
                    result_type = type_mapping[arg_types.reverse]
                elsif not arg_types.include?(PrimitiveType::Void) and receiver.type.is_primitive? and operand.type.is_primitive?
                    return TranslationResult.new(selector == :== ? "false" : "true", PrimitiveType::Bool)
                else
                    raise "Types #{receiver.type} and #{operand.type} not applicable for primitive operator #{selector.to_s}"
                end
            elsif logic_operators.include?(selector)
                # TODO: need proper implementation
                int_float = [PrimitiveType::Int, PrimitiveType::Float].to_set
                if selector == :'&&'
                    if (int_float + arg_types).size == 2
                        # Both are int/float
                        return operand
                    elsif operand.type == PrimitiveType::Bool and receiver.type == PrimitiveType::Bool
                        result_type = PrimitiveType::Bool
                    else
                        raise "Cannot handle types #{receiver.type} and #{operand.type} for primitive operator #{selector.to_s}"
                    end
                elsif selector == :'||'
                    if (int_float + arg_types).size == 2
                        # Both are int/float
                        return receiver
                    elsif operand.type == PrimitiveType::Bool and receiver.type == PrimitiveType::Bool
                        result_type = PrimitiveType::Bool
                    else
                        raise "Cannot handle types #{receiver.type} and #{operand.type} for primitive operator #{selector.to_s}"
                    end
                elsif selector == :& or selector == :| or selector == :^
                    type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => PrimitiveType::Bool,
                        [PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Int}
                        
                    if type_mapping.has_key?(arg_types)
                        result_type = type_mapping[arg_types]
                    else
                        raise "Types #{receiver.type} and #{operand.type} not applicable for primitive operator #{selector.to_s}"
                    end
                end
            end
            
            TranslationResult.new("(#{receiver.c_source} #{selector.to_s} #{operand.c_source})", result_type)
        else
            if receiver.type.to_ruby_type.singleton_methods.include?(("_ikra_c_" + selector.to_s).to_sym)
                # TODO: pass arguments
                receiver.type.to_ruby_type.send(("_ikra_c_" + selector.to_s).to_sym, receiver)
            else
                # TODO: set handle return value, pass arguments
                TranslationResult.new("#{receiver}.#{selector.to_s}()", PrimitiveType::Void)
            end
        end
    end
    
    def wrap_in_c_block(str)
        "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
    end
    
    def translate_begin(node, should_return = false)
        # BEGIN for loops, functions, etc. are handled separately
        # TODO: handle multiple statements in BEGIN
        
        if node.children.size > 1
            raise "Cannot handle multiple statements in BEGIN block"
        end
        
        translate_ast(node.children[0], should_return)
    end
    
    def translate_multi_begin_or_statement(node, last_returns = false)
        # This is for use in loops, functions, etc. only
        if node.type == :begin
            result_string = ""
            node.children[0...-1].each do |stmt|
                result_string += translate_statement(stmt).c_source
            end
            
            result_string += translate_statement(node.children.last, last_returns).c_source
            
            TranslationResult.new(result_string, PrimitiveType::Void)
        else
            result = translate_statement(node, last_returns)
            TranslationResult.new(result.c_source, PrimitiveType::Void)
        end
    end
    
    def translate_statement(node, should_return = false)
        if should_return and ExpressionStatements.include?(node.type)
            result = translate_ast(node, false)
            @symbol_table.ensure_defined(:"#", result.type) #
            TranslationResult.new("#{TempResultVariable} = #{result.c_source};\n", PrimitiveType::Void)
        elsif should_return
            TranslationResult.new(translate_ast(node, true).c_source + ";\n", PrimitiveType::Void)
        else
            TranslationResult.new(translate_ast(node, false).c_source + ";\n", PrimitiveType::Void)
        end
    end
end
