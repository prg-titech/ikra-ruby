require_relative "../../ast/nodes"
require_relative "../../ast/visitor"

module Ikra
    module Translator

        # This visitor inserts a synthetic method call whenever a parallel section should be
        # invoked, i.e.:
        # - The return value of the host section (must be an ArrayCommand-typed expression)
        # - When the content of an ArrayCommand-typed expression is accessed
        class ParallelSectionInvocationVisitor < AST::Visitor
            def visit_return_node(node)
                node.replace_child(
                    node.value,
                    AST::SendNode.new(receiver: node.value, selector: :__call__))
            end
        end

    end
end