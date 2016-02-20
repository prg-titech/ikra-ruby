require_relative "translator"
require_relative "compiler"
require_relative "types/primitive_type"

module ArrayCommand
    def [](index)
        if @result == nil
            execute
        end
        
        @result[index]
    end
end

class ArrayNewCommand
    include ArrayCommand
    
    def initialize(size, block)
        @size = size
        @block = block
    end
    
    def execute
        input_var = Translator::InputVariable.new(@block.parameters[0][1], PrimitiveType::Int, Translator::InputVariable::Index)
        @result = Translator.translate_block(@block, @size, [input_var]).execute
        @result
    end
    
    def size
        @size
    end
end

class ArrayMapCommand
    include ArrayCommand
    
    def initialize(target, block)
        @target = target
        @block = block
    end
    
    def execute
        input_var = Translator::InputVariable.new(@block.parameters[0][1], PrimitiveType::Int, Translator::InputVariable::Normal)
        proxy = Translator.translate_block(@block, size, [input_var])
        @result = proxy.execute(@target.execute)
        @result
    end
    
    def size
        @target.size
    end

    def translate
        compilation_result = @target.translate

        # TODO: how to handle size here?
        current_request = Compiler::CompilationRequest.new(block: @block, size: size)
        current_request.add_input_var(Translator::InputVariable.new(@block.parameters[0][1], UnknownType::Instance, Translator::InputVariable::PreviousFusion))
        compilation_result.merge_request!(current_request)

        compilation_result
    end
end

class ArrayIdentityCommand
    include ArrayCommand
    
    Block = Proc.new do |element|
        element
    end

    def initialize(target)
        @target = target
    end
    
    def execute
        @target
    end
    
    def size
        @target.size
    end

    def translate
        current_request = Compiler::CompilationRequest.new(block: Block, size: size)
        # TODO: check if all elements are of same type?
        current_request.add_input_var(Translator::InputVariable.new(Block.parameters[0][1], @target.first.class.to_ikra_type))
        Compiler.compile(current_request)
    end
end

class Array
    class << self
        def pnew(size, &block)
            ArrayNewCommand.new(size, block)
        end
    end
    
    def pmap(&block)
        ArrayMapCommand.new(to_command, block)
    end
    
    def to_command
        ArrayIdentityCommand.new(self)
    end
end

arr = [5, 7, 1, 3, 9, 9]
res = arr.pmap do |el|
    el * el
end
puts res.translate.full_source
