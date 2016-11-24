require "set"
require_relative "../translator/command_translator"
require_relative "../types/types"
require_relative "../type_aware_array"
require_relative "../parsing"
require_relative "../ast/nodes"
require_relative "../ast/lexical_variables_enumerator"
require_relative "../config/os_configuration"

Ikra::Configuration.check_software_configuration

module Ikra
    module Symbolic
        DEFAULT_BLOCK_SIZE = 256

        class Input
            attr_reader :command

            # Returns the access pattern of this input, e.g., `:tid` (single element, identified
            # by thread ID) or `:entire` (access to entire array is necessary).
            attr_reader :pattern

            def initialize(command:, pattern:)
                @command = command

                # Currently supported: :tid, :entire
                @pattern = pattern
            end
        end

        module ArrayCommand
            include Enumerable

            attr_reader :block_size

            # [Fixnum] Returns a unique ID for this command. It is used during name mangling in
            # the code generator to determine the name of array identifiers (and do other stuff?).
            attr_reader :unique_id

            # An array of commands that serve as input to this command. The number of input
            # commands depends on the type of the command.
            attr_reader :input

            @@unique_id  = 1

            def self.reset_unique_id
                @@unique_id = 1
            end

            def initialize
                super

                # Generate unique ID
                @unique_id = @@unique_id
                @@unique_id += 1
            end

            def [](index)
                if @result == nil
                    execute
                end
                
                @result[index]
            end

            def each(&block)
                next_index = 0

                while next_index < size
                    yield(self[next_index])
                    next_index += 1
                end
            end

            def pack(fmt)
                if @result == nil
                    execute
                end

                @result.pack(fmt)
            end

            def execute
                @result = Translator::CommandTranslator.translate_command(self).execute
            end
            
            def to_command
                self
            end
            
            def pmap(block_size: DEFAULT_BLOCK_SIZE, &block)
                ArrayMapCommand.new(self, block, block_size: block_size)
            end

            def pcombine(*others, block_size: Ikra::Symbolic::DEFAULT_BLOCK_SIZE, &block)
                return ArrayCombineCommand.new(self, others, block, block_size: block_size)
            end

            def +(other)
                return pcombine(other) do |a, b|
                    a + b
                end
            end

            def -(other)
                return pcombine(other) do |a, b|
                    a - b
                end
            end

            def *(other)
                return pcombine(other) do |a, b|
                    a * b
                end
            end

            def /(other)
                return pcombine(other) do |a, b|
                    a / b
                end
            end

            def |(other)
                return pcombine(other) do |a, b|
                    a | b
                end
            end

            def &(other)
                return pcombine(other) do |a, b|
                    a & b
                end
            end

            def ^(other)
                return pcombine(other) do |a, b|
                    a ^ b
                end
            end

            def pstencil(offsets, out_of_range_value, block_size: DEFAULT_BLOCK_SIZE, use_parameter_array: true, &block)
                ArrayStencilCommand.new(self, offsets, out_of_range_value, block, block_size: block_size, use_parameter_array: use_parameter_array)
            end

            # Returns a collection of the names of all block parameters.
            # @return [Array(Symbol)] list of block parameters
            def block_parameter_names
                block.parameters.map do |param|
                    param[1]
                end
            end

            # Returns the size (number of elements) of the result, after executing the parallel 
            # section.
            # @return [Fixnum] size
            def size
                raise NotImplementedError
            end

            def target
                raise NotImplementedError
            end

            # Returns the abstract syntax tree for a parallel section.
            def block_def_node
                # TODO: add caching for AST here
                parser_local_vars = block.binding.local_variables + block_parameter_names
                source = Parsing.parse_block(block, parser_local_vars)
                AST::BlockDefNode.new(
                    ruby_block: block,
                    body: AST::Builder.from_parser_ast(source))
            end

            # Returns a collection of lexical variables that are accessed within a parallel 
            # section.
            # @return [Hash{Symbol => Object}]
            def lexical_externals
                all_lexical_vars = block.binding.local_variables
                lexical_vars_enumerator = AST::LexicalVariablesEnumerator.new(all_lexical_vars)
                block_def_node.accept(lexical_vars_enumerator)
                accessed_variables = lexical_vars_enumerator.lexical_variables

                result = Hash.new
                for var_name in accessed_variables
                    result[var_name] = block.binding.local_variable_get(var_name)
                end

                result
            end

            # Returns a collection of external objects that are accessed within a parallel section.
            def externals
                lexical_externals.keys
            end

            protected

            # Returns the block of the parallel section.
            # @return [Proc] block
            def block
                raise NotImplementedError
            end
        end

        class ArrayNewCommand
            include ArrayCommand
            
            def initialize(size, block, block_size: DEFAULT_BLOCK_SIZE)
                super()

                @size = size
                @block = block
                @block_size = block_size

                # No input
                @input = []
            end
            
            def size
                @size
            end

            protected

            attr_reader :block
        end

        class ArrayMapCommand
            include ArrayCommand

            def initialize(target, block, block_size: DEFAULT_BLOCK_SIZE)
                super()

                @block = block
                @block_size = block_size

                # Read array at position `tid`
                @input = [Input.new(command: target, pattern: :tid)]
            end
            
            def size
                input.first.command.size
            end
            
            protected

            attr_reader :block
        end

        class ArrayCombineCommand
            include ArrayCommand

            def initialize(target, others, block, block_size: DEFAULT_BLOCK_SIZE)
                super()

                @block = block
                @block_size = block_size

                # Read array at position `tid`
                @input = [Input.new(command: target, pattern: :tid)] + others.map do |other|
                    Input.new(command: other, pattern: :tid)
                end
            end
            
            def size
                input.first.command.size
            end
            
            protected

            attr_reader :block
        end

        class ArrayStencilCommand
            include ArrayCommand

            attr_reader :offsets
            attr_reader :out_of_range_value
            attr_reader :use_parameter_array

            def initialize(target, offsets, out_of_range_value, block, block_size: DEFAULT_BLOCK_SIZE, use_parameter_array: true)
                super()

                @offsets = offsets
                @out_of_range_value = out_of_range_value
                @block = block
                @block_size = block_size
                @use_parameter_array = use_parameter_array

                # Read more than just one element, fall back to `:entire` for now
                @input = [Input.new(command: target, pattern: :entire)]
            end

            def size
                return input.first.command.size
            end

            def min_offset
                return offsets.min
            end

            def max_offset
                return offsets.max
            end

            protected

            attr_reader :block
        end

        class ArraySelectCommand
            include ArrayCommand

            def initialize(target, block)
                super

                @block = block

                # One element per thread
                @input = [Input.new(command: target, pattern: :tid)]
            end
            
            # how to implement SELECT?
            # idea: two return values (actual value and boolean indicator as struct type)
        end

        class ArrayIdentityCommand
            include ArrayCommand
            
            attr_reader :target

            DUMMY_BLOCK = Proc.new do |element|
                element
            end

            @@unique_id = 1

            def initialize(target)
                super()

                # Ensure that base array cannot be modified
                target.freeze

                # One thread per array element
                @input = [Input.new(command: target, pattern: :tid)]
            end
            
            def execute
                input.first.command
            end
            
            def size
                input.first.command.size
            end

            # Returns a collection of external objects that are accessed within a parallel section. This includes all elements of the base array.
            def externals
                lexical_externals.keys + input.first.command
            end

            def base_type
                # TODO: add caching (`input` is frozen)
                type = Types::UnionType.new

                input.first.command.each do |element|
                    type.add(element.class.to_ikra_type)
                end

                type
            end

            protected

            def block
                DUMMY_BLOCK
            end
        end
    end
end

class Array
    class << self
        def pnew(size, block_size: Ikra::Symbolic::DEFAULT_BLOCK_SIZE, &block)
            Ikra::Symbolic::ArrayNewCommand.new(size, block, block_size: block_size)
        end
    end
    
    def pmap(block_size: Ikra::Symbolic::DEFAULT_BLOCK_SIZE, &block)
        Ikra::Symbolic::ArrayMapCommand.new(to_command, block, block_size: block_size)
    end
    

    def pcombine(*others, block_size: Ikra::Symbolic::DEFAULT_BLOCK_SIZE, &block)
        return Ikra::Symbolic::ArrayCombineCommand.new(to_command, others, block, block_size: block_size)
    end

    alias_method :old_plus, :+
    alias_method :old_minus, :-
    alias_method :old_mul, :*
    alias_method :old_or, :|
    alias_method :old_and, :&

    def +(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            return pcombine(other) do |a, b|
                a + b
            end
        else
            return self.old_plus(other)
        end
    end

    def -(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            return pcombine(other) do |a, b|
                a - b
            end
        else
            return self.old_minus(other)
        end
    end
    
    def *(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            return pcombine(other) do |a, b|
                a * b
            end
        else
            return self.old_mul(other)
        end
    end

    def /(other)
        return pcombine(other) do |a, b|
            a / b
        end
    end

    def |(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            return pcombine(other) do |a, b|
                a | b
            end
        else
            return self.old_or(other)
        end
    end

    def &(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            return pcombine(other) do |a, b|
                a & b
            end
        else
            return self.old_and(other)
        end
    end

    def ^(other)
        return pcombine(other) do |a, b|
            a ^ b
        end
    end

    def pstencil(offsets, out_of_range_value, block_size: Ikra::Symbolic::DEFAULT_BLOCK_SIZE, use_parameter_array: true, &block)
        Ikra::Symbolic::ArrayStencilCommand.new(to_command, offsets, out_of_range_value, block, block_size: block_size, use_parameter_array: use_parameter_array)
    end

    def to_command
        Ikra::Symbolic::ArrayIdentityCommand.new(self)
    end
end

