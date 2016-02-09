require_relative "types/primitive_type"
require_relative "types/ruby_extension"
require_relative "scope"
require_relative "parsing"
require "set"
require "pp"
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
    
    def self.translate_block(block, input_types = [])
        instance = self.new
        instance.symbol_table.push_frame
        
        local_variables = []
        
        # what if variables are passed to the kernel and changed in-place?
        # we can implement locking or reduce-style merge for primitive arithmetic operators
        block.binding.local_variables.each do |var|
            instance.symbol_table.ensure_defined(var, block.binding.local_variable_get(var).class.to_ikra_type)
            local_variables.push(var)
        end
        
        (0..input_types.size - 1).each do |index|
            variable_name = block.parameters[index][1]
            
            # how do we pass in the values?
            instance.symbol_table.define_shadowed(variable_name, input_types[index])
            local_variables.push(variable_name)
        end
        
        ast = Parsing.parse(block, local_variables)
        result = instance.translate_function(ast)
        
        instance.symbol_table.pop_frame
        
        result
    end
    
    def initialize
        @symbol_table = Scope.new
    end
    
    def symbol_table
        @symbol_table
    end
    
    def translate_function(node)
        result = nil
        
        @symbol_table.new_frame do
            result = translate_multi_begin_or_statement(node)
            result = variable_definitions + result.c_source
        end
        
        result
    end
    
    def variable_definitions
        result_string = @symbol_table.last.map do |name, type|
            type.to_c_type + " " + name.to_s
        end.join(";\n")
        
        if result_string.size > 0
            result_string += ";\n"
        end
        
        result_string
    end
    
    def translate_ast(node)
        send("translate_#{node.type.to_s}".to_sym, node)
    end
    
    def translate_int(node)
        TranslationResult.new(node.children[0], PrimitiveType::Int)
    end
    
    def translate_float(node)
        TranslationResult.new(node.children[0], PrimitiveType::Float)
    end
    
    def translate_bool(node)
        TranslationResult.new(node.children[0], PrimitiveType::Bool)
    end
    
    def translate_lvar(node)
        variable_name = node.children[0]
        TranslationResult.new(variable_name.to_s, @symbol_table.get_type(variable_name))
    end
    
    def translate_lvasgn(node)
        variable_name = node.children[0]
        value = translate_ast(node.children[1])
        
        @symbol_table.ensure_defined(variable_name, value.type)
        TranslationResult.new("#{variable_name} = #{value.c_source}", value.type)
    end
    
    def translate_if(node)
        # TODO: handle inline IF expressions
        
        condition = translate_ast(node.children[0])
        if condition.type != PrimitiveType::Bool
            raise "Expected boolean expression for IF condition"
        end
        
        result_string = "if (#{condition.c_string})\n" + wrap_in_c_block(translate_multi_begin_or_statement(node.children[1]).c_source)
        
        if (node.children.size == 3)
            result_string += "else\n" + wrap_in_c_block(translate_multi_begin_or_statement(node.children[2]).c_source)
        end
        
        TranslationResult.new(result_string, PrimitiveType::Void)
    end
    
    def translate_for(node)
        if node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :irange
            # TODO: this is the only kind of for loop we can handle right now
            variable_name = node.children[0].children[0]
            range = extract_begin_single_statement(node.children[1])
            range_from = translate_ast(range.children[0])
            range_to = translate_ast(range.children[1])
            
            if range_from.type != :int or range_to.type != :int
                raise "Expected range with only integers"
            end
            
            @symbol_table.ensure_defined(variable_name, PrimitiveType::Int)
            
            result_string = "for (#{variable_name} = (#{range_from.c_source}); #{variable_name} <= (#{range_to.c_source}); #{variable_name}++)\n"
               + wrap_in_c_block(translate_multi_begin_or_statement(node.children[2]).c_source)
               
            TranslationResult.new(result_string, PrimitiveType::Void)
        end
        
    end
    
    def extract_begin_single_statement(node)
        next_node = node
        while next_node.type == :begin
            if next_node.children.size != 1
                raise "Begin node contains more than one statement"
            end
            
            next_node = next_node.children[0]
        end
        
        next_node
    end
    
    def translate_send(node)
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
                TranslationResult.new(receiver.type.to_ruby_type.send(("_ikra_c_" + selector.to_s).to_sym, receiver.c_source))
            else
                # TODO: set handle return value, pass arguments
                TranslationResult.new("#{receiver}.#{selector.to_s}()", PrimitiveType::Void)
            end
        end
    end
    
    def wrap_in_c_block(str)
        "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "}\n"
    end
    
    def translate_begin(node)
        # BEGIN for loops, functions, etc. are handled separately
        # TODO: handle multiple statements in BEGIN
        
        if node.children.size > 1
            raise "Cannot handle multiple statements in BEGIN block"
        end
        
        translate_ast(node.children[0])
    end
    
    def translate_multi_begin_or_statement(node)
        # This is for use in loops, functions, etc. only
        
        if node.type == :begin
            result_string = ""
            node.children.each do |stmt|
                result_string += translate_ast(stmt).c_source + ";\n"
            end
            
            TranslationResult.new(result_string, PrimitiveType::Void)
        else
            result = translate_statement(node)
            TranslationResult.new(result.c_source, PrimitiveType::Void)
        end
    end
    
    def translate_statement(node)
        TranslationResult.new(translate_ast(node).c_source, PrimitiveType::Void)
    end
end

x = 1
y = 2
z = Proc.new do |a|
    a =x*x
end

puts Translator.translate_block(z)