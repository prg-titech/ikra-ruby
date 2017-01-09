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
                raise NotImplementedError.new
            end

            # Returns the abstract syntax tree for this section.
            def block_def_node
                if @ast == nil
                    parser_local_vars = block.binding.local_variables + block_parameter_names
                    source = Parsing.parse_block(block, parser_local_vars)
                    @ast = AST::BlockDefNode.new(
                        ruby_block: block,      # necessary to get binding
                        body: AST::Builder.from_parser_ast(source))
                end

                return @ast
            end
        end

        def self.host_section(*section_input, &block)
            return ArrayHostSectionCommand.new(*section_input, &block)
        end
    end
end
