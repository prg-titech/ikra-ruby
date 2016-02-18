require_relative "translator"
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
        input_var = Translator::InputVariable.new(@block.parameters[0][1], PrimitiveType::Int, true)
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
end

class ArrayIdentityCommand
    include ArrayCommand
    
    def initialize(target)
        @target = target
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
