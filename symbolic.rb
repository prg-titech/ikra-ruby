require "set"
require_relative "translator/compiler"
require_relative "types/primitive_type"
require_relative "types/class_type"
require_relative "types/union_type"
require_relative "type_aware_array"

module ArrayCommand
    def [](index)
        if @result == nil
            execute
        end
        
        @result[index]
    end
    
    def execute
        compilation_request = translate
        allocate(compilation_request)
        @result = compilation_request.execute
    end
    
    def to_command
        self
    end
    
    def pmap(&block)
        ArrayMapCommand.new(self, block)
    end
end

class ArrayNewCommand
    include ArrayCommand
    
    def initialize(size, block)
        @size = size
        @block = block
    end
    
    def translate
        current_request = Ikra::Translator::Compiler::CompilationRequest.new(block: @block, size: size)
        # TODO: check if all elements are of same type?
        current_request.add_input_var(Ikra::Translator::Compiler::InputVariable.new(@block.parameters[0][1], Ikra::Types::UnionType.create_int, Ikra::Translator::Compiler::InputVariable::Index))
        Ikra::Translator::Compiler.compile(current_request)
    end
    
    def size
        @size
    end
    
    def allocate(compilation_request)
        
    end
end

class ArrayMapCommand
    include ArrayCommand
    
    def initialize(target, block)
        @target = target
        @block = block
    end
    
    def size
        @target.size
    end

    def translate
        compilation_result = @target.translate

        current_request = Ikra::Translator::Compiler::CompilationRequest.new(block: @block, size: size)
        current_request.add_input_var(Ikra::Translator::Compiler::InputVariable.new(@block.parameters[0][1], Ikra::Types::UnionType.create_unknown, Ikra::Translator::Compiler::InputVariable::PreviousFusion))
        compilation_result.merge_request!(current_request)

        compilation_result
    end
    
    def allocate(compilation_request)
        @target.allocate(compilation_request)
    end
end

class ArraySelectCommand
    include ArrayCommand

    def initialize(target, block)
        @target = target
        @block = block
    end
    
    # how to implement SELECT?
    # idea: two return values (actual value and boolean indicator as struct type)
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
        current_request = Ikra::Translator::Compiler::CompilationRequest.new(block: Block, size: size)
        
        types = @target.all_types.map do |cls|
            cls.to_ikra_type
        end

        current_request.add_input_var(Ikra::Translator::Compiler::InputVariable.new(Block.parameters[0][1], Ikra::Types::UnionType.new(*types.to_set.to_a)))
        Ikra::Translator::Compiler.compile(current_request)
    end
    
    def allocate(compilation_request)
        compilation_request.allocate_input_array(@target)
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

#arr = [5, 7, 1, 3, 9, 9]
#res = arr.pmap do |el|
#    el * el
#end.pmap do |el|
#    el + 1
#end

#puts res.execute
