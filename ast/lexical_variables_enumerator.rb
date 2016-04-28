require "set"

module Ikra
    module AST

        # Visitor for determining the names all lexical variables that are accessed by a block (or method??)
        # TODO: does a method have access to lexical variables?
        class LexicalVariablesEnumerator < Visitor
            attr_reader :lexical_variables

            def initialize(lexical_var_names)
                @lexical_var_names = lexical_var_names
                @lexical_variables = Set.new
            end

            def add_lvar_access(identifier)
                if @lexical_var_names.include?(identifier)
                    @lexical_variables.add(identifier)
                end
            end

            def visit_lvar_read_node(node)
                add_lvar_access(node.identifier)
            end

            def visit_lvar_write_node(node)
                add_lvar_access(node.identifier)
                super(node)
            end
        end
    end
end