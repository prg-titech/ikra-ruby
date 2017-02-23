require_relative "../ast/host_section_builder"

module Ikra
    module Symbolic
        # The return value of a host section. For the moment, every host section can only have
        # one result.
        class ArrayHostSectionCommand
            include ArrayCommand

            attr_reader :block
            attr_reader :section_input

            def initialize(*section_input, &block)
                @block = block
                @section_input = section_input
            end

            def size
                execute
                return @result.size
            end

            # Returns the abstract syntax tree for this section.
            def block_def_node
                if @ast == nil
                    # Get array of block parameter names
                    block_params = block.parameters.map do |param|
                        param[1]
                    end

                    parser_local_vars = block.binding.local_variables + block_params
                    source = Parsing.parse_block(block, parser_local_vars)
                    @ast = AST::BlockDefNode.new(
                        parameters: block_params,
                        ruby_block: block,      # necessary to get binding
                        body: AST::HostSectionBuilder.from_parser_ast(source))
                end

                return @ast
            end

            def command_translator_class
                return Translator::HostSectionCommandTranslator
            end
        end

        # An array that that is referenced using C++/CUDA expressions. Such an array does not
        # necessarily have to be present in the Ruby interpreter. Its size does also not have to
        # be known at compile time.
        class ArrayInHostSectionCommand
            include ArrayCommand
            
            attr_accessor :target
            attr_accessor :base_type

            def initialize(target, base_type, block_size: DEFAULT_BLOCK_SIZE)
                super(block_size: block_size)

                if base_type == nil
                    raise AssertionError.new("base_type missing")
                end

                # One thread per array element
                @input = [SingleInput.new(command: target, pattern: :tid)]
                @base_type = base_type
            end
            
            def size
                # Size is not known at compile time. Return a source code string here.
                return "#{input.first.command}->size()"
            end

            # TODO: Support multiple dimensions
        end

        def self.host_section(*section_input, &block)
            return ArrayHostSectionCommand.new(*section_input, &block)
        end
    end
end
