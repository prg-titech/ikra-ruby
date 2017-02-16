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

            def visit_symbol_node(node)
                return node.value
            end

            def visit_string_node(node)
                return node.value
            end

            def visit_array_node(node)
                return node.values.map do |value|
                    value.accept(self)
                end
            end

            def visit_const_node(node)
                return node.find_behavior_node.binding.eval(node.identifier)
            end

            def visit_hash_node(node)
                result = {}

                node.hash.each do |key, value|
                    result[key.accept(self)] = value.accept(self)
                end

                return result
            end

            def visit_send_node(node)
                receiver = node.receiver.accept(self)
                arguments = node.arguments.map do |arg| arg.accept(self) end

                if node.block_argument == nil
                    return receiver.send(node.selector, *arguments)
                else
                    # TODO: Implement
                    block = receiver.block_argument.accept(self)
                    return receiver.send(node.selector, *arguments, &block)
                end
            end
        end
    end
end
