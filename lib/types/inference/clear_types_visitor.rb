module Ikra
    module TypeInference
        class ClearTypesVisitor < AST::Visitor
            def visit_node(node)
                if node.respond_to?(:get_type)
                    node.get_type.clear!
                end
            end
        end
    end
end