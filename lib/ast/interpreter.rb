module Ikra
    module AST
        # Executes this AST in the Ruby interpreter. Not possible for all nodes.
        class Interpreter < Visitor
            def self.interpret(node)
                return node.accept(self.new)
            end

            def visit_int_node(node)
                return node.value
            end

            def visit_float_node(node)
                return node.value
            end

            def visit_bool_node(node)
                return node.value
            end

            def visit_nil_node(node)
                return nil
            end

            def visit_array_node(node)
                return node.values.map do |value|
                    value.accept(self)
                end
            end

            def visit_send_node(node)
                receiver = node.receiver.accept(self)
                arguments = node.arguments.map do |arg| arg.accept(self) end

                if node.block_argument == nil
                    return receiver.send(node.receiver, *arguments)
                else
                    # TODO: Implement
                    block = receiver.block_argument.accept(self)
                    return receiver.send(node.receiver, *arguments, &block)
                end
            end
        end
    end
end
