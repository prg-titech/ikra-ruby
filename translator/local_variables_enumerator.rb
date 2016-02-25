require_relative "../ast/nodes"
require_relative "../ast/visitor"
require_relative "../types/type_inference"

module Ikra
    module Translator
        class LocalVariablesEnumerator < AST::Visitor
            def initialize
                @vars = {}
            end

            def add_local_var(var, types)
                @vars[var] = types
            end

            def local_variables
                @vars
            end

            def visit_lvar_read_node(node)
                add_local_var(node.identifier, node.get_types)
            end
            
            def visit_lvar_write_node(node)
                add_local_var(node.identifier, node.get_types)
                node.value.accept(self)
            end
        end
    end
end