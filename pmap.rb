require "parser/current"
require "sourcify"
require "pp"
require "ripper"
require "tempfile"
require "ffi"

# OPTIMIZATIONS
# - retrieve max. number of registers, then ?

class ArrayCommand
    def [](index)
        return result()[index]
    end
    
    def pmap(&block)
        return ArrayMapCommand.new(self, &block)
    end
    
    def pselect(&block)
        return ArraySelectCommand.new(self, &block)
    end
    
    def parser
        return Parser::CurrentRuby
    end
    
    protected
    
    def result
        @cached ||= execute()
        return @cached
    end
    
    class Translator
        VAR_DECL_SYMBOL = :DECL_ONLY
        
        def parse_block(block)
            parser = Parser::CurrentRuby.default_parser
            block.binding.local_variables.each do |var|
                parser.static_env.declare(var)
            end
            block.parameters.map do |param|
                parser.static_env.declare(param[1])
            end
            
            parser_source = Parser::Source::Buffer.new('(string)', 1)
            parser_source.source = block.to_source(strip_enclosure: true)
            
            return parser.parse(parser_source)
        end
        
        def translate(block, func_name, command)
            externs = {}
            block.binding.local_variables.each do |var|
                externs[var] = block.binding.local_variable_get(var).class.to_ikra_type
            end
            
            # No parameter for Array.new
            # TODO: make type variable
            externs[block.parameters[0][1]] = SymbolTable::PrimitiveType::INT
            
            @symbol_table = SymbolTable.new
            @symbol_table.merge!(externs)
            @all_variables = []
            @result_param = "_result_"
            
            ast = parse_block(block)
            @functions = {func_name => ast}
            result = ""
            
            while not @functions.empty? do
                pair = @functions.first
                @functions.delete(pair[0])
                block_translated = translate_block(pair[1], true)
                selected_externs = externs.to_a
                    .select do |var| @all_variables.include?(var[0]) end
                    .reject do |var| var[0] == block.parameters[0][1] end   # remove input var
                @kernel_selected_externs ||= selected_externs
                externs_decl = selected_externs
                    .map do |pair| pair[1].c_type_name + " " + pair[0].to_s end
                    .join(", ")
                
                if @return_type != nil
                    if externs_decl == nil
                        # TODO: check this part
                        externs_decl = @return_type.c_type_name + "* " + @result_param
                    else
                        externs_decl += ", " + @return_type.c_type_name + "* " + @result_param
                    end
                end
                
                function_block_string_split = block_translated.split("\n")
                function_block_string = function_block_string_split[0] + "\n    " + command.assign_input + function_block_string_split.drop(1).join("\n")
                result += "__global__ void #{pair[0]}(#{externs_decl})\n" + function_block_string
            end
            
            return TranslationResult.new(result, @return_type, @kernel_selected_externs)
        end
        
        def translate_block(node, should_write_back = false)
            content = nil
            
            if node.type == :begin
                children = node.children
                content = children[0...-1].map do |s| translate_ast(s, false) end.join(";\n") + ";\n"
                content += translate_ast(children.last, should_write_back) + ";\n"
            else node.type != :begin
                # Only one statement
                content = translate_ast(node, should_write_back) + "\n"
            end
            
            
            indented = content.split("\n").map do |line| "    " + line end.join("\n")
            return "{\n#{indented}\n}\n"
        end
        
        BINARY_ARITHMETIC_OPERATORS = [:+, :-, :*, :/, :%]
        BINARY_COMPARISON_OPERATORS = [:==, :>, :<, :>=, :<=, :!=]
        BINARY_LOGICAL_OPERATORS = [:|, :"||", :&, :"&&", :^]
        
        class TranslationResult
            def initialize(translation, type, externs = {})
                @translation = translation
                @type = type
                @externs = externs
            end
            
            def translation
                return @translation
            end
            
            def type
                return @type
            end
            
            def to_str
                return translation
            end
            
            def to_s
                return to_str
            end
            
            def +(other)
                return to_str + other.to_str
            end
            
            def externs
                return @externs
            end
        end
        
        def maybe_write_back(value, should_write_back)
            if should_write_back
                @return_type ||= value.type
                
                if @return_type != value.type
                    raise "ERROR: mismatch in return value types"
                end
                
                TranslationResult.new(should_write_back ? "_result_[blockIdx.x] = " + value.to_s : value.to_s, value.type)
            else
                value
            end
        end
        
        def translate_ast(node, should_write_back = false)
            # if node.should_return: prepend with "return"
            return_value = case node.type
                when :int
                    value = node.children[0].to_s
                    maybe_write_back(TranslationResult.new(value, SymbolTable::PrimitiveType::INT), should_write_back)
                when :float
                    value = node.children[0].to_s
                    maybe_write_back(TranslationResult.new(value, SymbolTable::PrimitiveType::FLOAT), should_write_back)
                when :lvar
                    var_name = node.children[0].to_s
                    @all_variables.push(var_name.to_sym)
                    maybe_write_back(TranslationResult.new(var_name, @symbol_table[node.children[0].to_sym]), should_write_back)
                when :lvasgn
                    if (node.children.size > 1 && node.children[1].type == :sym && node.children[1].children[0] == Translator::VAR_DECL_SYMBOL)
                        # TODO: find better way to do this
                        TranslationResult.new("", SymbolTable::PrimitiveType::VOID)
                    else
                        var_name = node.children[0].to_sym
                        result = ""
                        result_type = nil
                        
                        if (@symbol_table.has_key?(var_name))
                            if (node.children.size > 1)
                                # This is an assignment
                                value = translate_ast(node.children[1])
                                @symbol_table.assert(var_name, value.type)
                                maybe_write_back(TranslationResult.new(var_name.to_s + " = " + value, value.type), should_write_back)
                            else
                                # TODO: check if this is correct
                                # Treat as variable read
                                maybe_write_back(TranslationResult.new(var_name, @symbol_table[var_name]), should_write_back)
                            end
                        else
                            # TODO: implement write back
                            if (node.children.size > 1)
                                # This is an assignment
                                value = translate_ast(node.children[1])
                                @symbol_table.insert(var_name, value.type)
                                TranslationResult.new(value.type.c_type_name + " " + var_name.to_s + " = " + value, value.type)
                            else
                                # TODO: infer type
                                @symbol_table.insert(var_name, SymbolTable::PrimitiveType::INT)
                                TranslationResult.new(value.type.c_type_name + " " + var_name.to_s, SymbolTable::PrimitiveType::VOID)
                            end
                        end
                    end
                when :send
                    receiver = node.children[0] == nil ? TranslationResult.new("", SymbolTable::PrimitiveType::VOID) : translate_ast(node.children[0])

                    if BINARY_ARITHMETIC_OPERATORS.include?(node.children[1])
                        operand = translate_ast(node.children[2])
                        value = "(" + receiver + " " + node.children[1].to_s + " " + operand + ")"
                        
                        if receiver.type.is_primitive? and operand.type.is_primitive?
                            # implicit type conversions allowed
                            
                            if [receiver.type, operand.type].include?(SymbolTable::PrimitiveType::BOOL)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL"
                            elsif [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::VOID)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to VOID"
                            end
                            
                            if receiver.type == SymbolTable::PrimitiveType::DOUBLE or operand.type == SymbolTable::PrimitiveType::DOUBLE
                                type = SymbolTable::PrimitiveType::DOUBLE
                            elsif receiver.type == SymbolTable::PrimitiveType::FLOAT or operand.type == SymbolTable::PrimitiveType::FLOAT
                                type = SymbolTable::PrimitiveType::FLOAT
                            elsif receiver.type == SymbolTable::PrimitiveType::INT or operand.type == SymbolTable::PrimitiveType::INT
                                type = SymbolTable::PrimitiveType::INT
                            end
                            
                            maybe_write_back(TranslationResult.new(value.to_str, type), should_write_back)
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                    elsif BINARY_COMPARISON_OPERATORS.include?(node.children[1])
                        operand = translate_ast(node.children[2])
                        value = "(" + receiver + " " + node.children[1].to_s + " " + operand + ")"
                        
                        if receiver.type.is_primitive? and operand.type.is_primitive?
                            # implicit type conversions allowed
                            
                            if [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::BOOL)
                                if [receiver.type,  operand.type].uniq.size == 2
                                    if node.children[1] == :==
                                        # comparing objects of differnt types for identity
                                        value = "false"
                                    elsif node.children[1] == :!=
                                        value = "true"
                                    else
                                        raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL and other type"
                                    end
                                else
                                    if not [:==, :!=].include?(node.children[1])
                                        raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL"
                                    end
                                end
                            elsif [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::VOID)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to VOID"
                            end
                            
                            maybe_write_back(TranslationResult.new(value.to_str, SymbolTable::PrimitiveType::BOOL), should_write_back)
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                    elsif BINARY_LOGICAL_OPERATORS.include?(node.children[1])
                        operand = translate_ast(node.children[2])
                        operator = node.children[1]
                        value = "(" + receiver + " " + operator.to_s + " " + operand + ")"
                        int_float = [SymbolTable::PrimitiveType::INT, SymbolTable::PrimitiveType::FLOAT]
                        type = nil
                        
                        if int_float.include?(receiver.type) and int_float.include?(operand.type)
                            if operator == :"||"
                                value = receiver
                                type = receiver.type
                            elsif operator == :"&&"
                                value = operand
                                type = operand.type
                            end
                        elsif receiver.type == SymbolTable::PrimitiveType::BOOL and operand.type == SymbolTable::PrimitiveType::BOOL
                            type = SymbolTable::PrimitiveType::BOOL
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                        
                        maybe_write_back(TranslationResult.new(value.to_str, type), should_write_back)
                    elsif receiver.type.ruby_type.singleton_methods.include?(("_ikra_c_" + node.children[1].to_s).to_sym)
                        symbol = ("_ikra_c_" + node.children[1].to_s).to_sym
                        maybe_write_back(receiver.type.ruby_type.send(symbol, receiver.to_s), should_write_back)
                    else
                        if node.children[0] != nil
                            receiver += "."
                        end
                        
                        value = receiver + node.children[1].to_s + "(" + node.children[2..-1].map { |c| translate_ast(c) }.join(", ") + ")"
                        # TODO: type inference for objects
                        maybe_write_back(TranslationResult.new(value, SymbolTable::PrimitiveType::INT), should_write_back)
                    end
                when :begin
                    if node.children.size == 1
                        translate_ast(node.children[0], should_write_back)
                    else
                        funcId = nextFuncId
                        @functions[funcId] = node
                        funcId + "()"
                    end
                when :if
                    condition = translate_ast(node.children[0])
                    
                    if (condition.type != SymbolTable::PrimitiveType::BOOL)
                        raise "ERROR: only BOOL allowed in condition but saw #{condition.type}"
                    end
                    
                    result = "if (#{condition.to_str})\n" + translate_block(node.children[1], should_write_back)
                    
                    if node.children.size > 2 && node.children[2] != nil
                        TranslationResult.new(result + "else\n" + translate_block(node.children[2], should_write_back), SymbolTable::PrimitiveType::VOID)
                    else
                        TranslationResult.new(result, SymbolTable::PrimitiveType::VOID)
                    end
                when :args
                    ""
                when :for
                    # TODO: implement write back
                    variable = translate_ast(node.children[0])
                    variable_name = node.children[0].children[0]
                    
                    if node.children[1].type == :begin && node.children[1].children[0].type == :irange
                        # for loop interating over range
                        range_expr = node.children[1].children[0]
                        range_from = translate_ast(range_expr.children[0])
                        range_to = translate_ast(range_expr.children[1])
                        
                        if variable.type != range_from.type || range_from.type != range_to.type
                            raise "ERROR: type mismatch in iterator variable of loop"
                        end
                        
                        result = "for (#{variable.to_s} = #{range_from}; #{variable_name} <= #{range_to}; #{variable_name}++)\n" + translate_block(node.children[2], false)
                        
                        TranslationResult.new(result, SymbolTable::PrimitiveType::VOID)
                    else
                        raise "FOR not implemented"
                    end
                when :break
                    TranslationResult.new("break", SymbolTable::PrimitiveType::VOID)
                else
                    puts "MISSING: " + node.type.to_s
                    ""
            end
            
            return return_value
        end
        
        def next_id
            @next_id ||= 0
            @next_id += 1
            return @next_id
        end
        
        def nextFuncId
            return "func_" + next_id.to_s
        end
        
        def parser
            return Parser::CurrentRuby
        end
        
        class SymbolTable < Hash
            class RubyObject
                RUBYOBJECT = self.new
                
                def c_type_name
                    return "RubyObject"
                end
            end
            
            class PrimitiveType
                def initialize(type, c_type, ruby_type)
                    @type = type
                    @c_type = c_type
                    @ruby_type = ruby_type
                end
                
                def c_type_name
                    return @c_type
                end
                
                def is_primitive?
                    return true
                end
                
                def ruby_type
                    return @ruby_type
                end

                INT = self.new(:int, "int", Fixnum)
                FLOAT = self.new(:float, "float", Float)
                DOUBLE = self.new(:double, "double", Float)
                BOOL = self.new(:bool, "bool", TrueClass)
                VOID = self.new(:void, "void", nil)
            end
            
            def assert(symbol, type)
                if type != self[symbol]
                    raise "ERROR: #{symbol} has type #{self[symbol]}, but requires #{type}"
                end
            end
            
            def insert(symbol, type)
                if self.has_key?(symbol)
                    raise "ERROR: #{symbol} already defined"
                end
                
                self[symbol] = type
            end
            
            def [](symbol)
                if self.has_key?(symbol)
                    return self.fetch(symbol)
                else
                    raise "ERROR: #{symbol} expected but not found in symbol table"
                end
            end
        end
    end
end

class Object
    def self.to_ikra_type
        return ArrayCommand::Translator::SymbolTable::RubyObject::RUBYOBJECT
    end
end

class TrueClass
    def self.to_ikra_type
        return ArrayCommand::Translator::SymbolTable::PrimitiveType::BOOL
    end
    
    def self.ffi_type_symbol
        :bool
    end
end

class FalseClass
    def self.to_ikra_type
        return ArrayCommand::Translator::SymbolTable::PrimitiveType::BOOL
    end
    
    def self.ffi_type_symbol
        :bool
    end
end

class Fixnum
    def self.to_ikra_type
        return ArrayCommand::Translator::SymbolTable::PrimitiveType::INT
    end
    
    def self._ikra_c_to_f(receiver)
        ArrayCommand::Translator::TranslationResult.new("(float) (#{receiver})", Float.to_ikra_type)
    end
    
    def self.ffi_type_symbol
        :int
    end
end

class Float
    def self.to_ikra_type
        return ArrayCommand::Translator::SymbolTable::PrimitiveType::FLOAT
    end
    
    def self._ikra_c_to_f(receiver)
        ArrayCommand::Translator::TranslationResult.new("(#{receiver})", to_ikra_type)
    end
    
    def self.ffi_type_symbol
        :float
    end
end

class ArrayMapCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().map(&@block)
    end
    
    def source
        ast = self.parser.parse(@block.to_source(strip_enclosure: true))
        #ast = Ripper.sexp(@block.to_source(strip_enclosure: true))
        pp ast
        puts ArrayCommand::Translator.new.translate(ast, "main")
        #pp ast.children[0]
        # use @block.binding to get local variables in context
    end
    
    def size
        return @target.size
    end
end

class ArraySelectCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().select(&@block)
    end
    
    def source
        puts self.parser.parse(@block.to_source(strip_enclosure: true))
    end
    
    def size
        return @target.size
    end
end

class ArrayIdentityCommand < ArrayCommand
    def initialize(target)
        @target = target
    end
    
    def execute
        return @target
    end
    
    def size
        return @target.size
    end
end

class ArrayNewCommand < ArrayCommand
    def initialize(size, &block)
        @size = size
        @block = block
    end
    
    def source
        ArrayCommand::Translator.new.translate(@block, "main_kernel", self)
    end
    
    def size
        return @size
    end
    
    def compile_and_run
        translation = source
        block_source = translation.to_s
        result_type = translation.type
        externs_decl = translation.externs
            .map do |pair| pair[1].c_type_name + " " + pair[0].to_s end
            .join(", ")
        passed_params = (translation.externs + [["device_result", ""]])
            .map do |pair| pair[0] end
            .join(", ")
        content = block_source + """
extern \"C\" __declspec(dllexport) int launch_kernel(#{externs_decl})
{
    #{result_type.c_type_name} * host_result = (#{result_type.c_type_name}*) malloc(sizeof(#{result_type.c_type_name}) * #{size});
    #{result_type.c_type_name} * device_result;
    
    cudaMalloc(&device_result, sizeof(#{result_type.c_type_name}) * #{size});
    
    dim3 dim_grid(#{size}, 1, 1);
    dim3 dim_block(1, 1, 1);
    main_kernel<<<dim_grid, dim_block>>>(#{passed_params});
    
    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, sizeof(#{result_type.c_type_name}) * #{size}, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    
    return 1;
}
        """
        
        puts content
        
        file = Tempfile.new(["ikra_kernel", ".cu"])
        file.write(content)
        file.close
        puts %x(nvcc -o #{file.path}.dll --shared #{file.path})
        
        
        ffi_wrapper = Module.new
        ffi_wrapper.extend(FFI::Library)
        ffi_wrapper.ffi_lib(file.path + ".dll")
        
        attached_params = translation.externs.map do |pair| pair[1].ruby_type.ffi_type_symbol end
        ffi_wrapper.attach_function(:launch_kernel, attached_params, :int)
        
        binding_variables = translation.externs.map do |pair| 
            @block.binding.local_variable_get(pair[0].to_sym) 
        end
        ffi_wrapper.launch_kernel(*binding_variables)
    end
    
    def assign_input
        "int #{@block.parameters[0][1].to_s} = blockIdx.x;\n"
    end
end

class Array
    def pmap(&block)
        return ArrayIdentityCommand.new(self).pmap(&block)
    end
    
    def self.pnew(size, &block)
        return ArrayNewCommand.new(size, &block)
    end
end


#x= 1000
#p = Proc.new do
#    x+1+bla()
#end
#puts p.call()
#puts p.binding.local_variables


#puts ([1,2,3].pmap do |x| x+1 end.pmap do |x| x *10 end.pselect do |x| x > 20 end).source()
#a=[1,2,3].pmap do |x| x + 2 end
#    .pmap { |x | x=0; x * 10 * foo(1,2,3)
#    Array.new {|y| y + x} }
#a.source

#a = [1,2,3].pmap do |x| 
#    x + 2
#    y = x + 2 * 5.0
#    y = x * 2 + 5.0
#    y = y + 2
#    array.map do |ppp|
#    
#    end
#    
#    foo(begin
#        puts 123
#        4
#    end)
#    
#    if (y==y) == (y==y)
#        puts 123
#    else
#        puts x
#    end
#    9+0
#    x.foo().bar()
#end
#a.source

magnify = 1.0
hx_res = 500
hy_res = 500
iter_max = 100
mandel = Array.pnew(hx_res * hy_res) do |i|
    hx = i % hx_res
    hy = i / hx_res
    
    cx = (hx.to_f / hx_res.to_f - 0.5) / magnify*3.0 - 0.7
    cy = (hy.to_f / hy_res.to_f - 0.5) / magnify*3.0
    
    x = 0.0
    y = 0.0
    
    for i in 0..iter_max
        xx = x*x - y*y + cx
        y = 2.0*x*y + cy
        x = xx
        
        if x*x + y*y > 100
            i = 101
            break
        end
    end
    
    if i == 101
        1
    else
        0
    end
end

puts mandel.compile_and_run

