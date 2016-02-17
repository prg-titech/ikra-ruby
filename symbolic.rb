require_relative "translator"
require_relative "types/primitive_type"

class ArrayNewCommand
    def initialize(size, block)
        @size = size
        @block = block
    end
    
    def execute
        @result = Translator.translate_block(@block, @size, [PrimitiveType::Int]).execute
        @result
    end
    
    def size
        @size
    end
    
    def [](index)
        if @result == nil
            execute
        end
        
        @result[index]
    end
end

class ArrayMapCommand
    def initialize(target, block)
        @target = target
        @block = block
    end
end

class ArrayIdentityCommand
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
