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
                node.parent.replace_child(node, AST::SendNode.new(
                    receiver: node,
                    selector: :__call__))
                # TODO: Do we have to set the type as well here?
            end
        end

    end
end