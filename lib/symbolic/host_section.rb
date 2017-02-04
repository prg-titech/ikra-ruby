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

        def self.host_section(*section_input, &block)
            return ArrayHostSectionCommand.new(*section_input, &block)
        end
    end
end
