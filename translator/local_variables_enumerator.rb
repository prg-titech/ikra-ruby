require_relative "../ast/nodes"
require_relative "../ast/visitor"
require_relative "../types/type_inference"

module Ikra
    module Translator
        class LocalVariablesEnumerator < AST::Visitor
            def initialize
                @vars = {}
            end

            def add_local_var(var, type)
                @vars[var] = type
            end

            def local_variables
                @vars
            end

            def visit_lvar_read_node(node)
                add_local_var(node.identifier, node.get_type)
            end
            
            def visit_lvar_write_node(node)
                add_local_var(node.identifier, node.get_type)
                super(node)
            end

            def visit_for_node(node)
                add_local_var(node.iterator_identifier, Types::UnionType.create_int)
                super(node)
            end
        end
    end
end