module Ikra
    module AST
        class HostSectionBuilder < Builder

            protected

            def translate_block(node)
                if node.children[0].type != :send
                    raise AssertionError.new("Unknown AST construct: Send node expected")
                end

                send_node = node.children[0]
                built_block_node = BlockDefNode.new(
                    body: translate_node(node.children[2]),
                    parameters: translate_node(node.children[1]),
                    ruby_block: nil)

                if send_node.children[0] == nil && send_node.children[1] == :proc
                    # Defining a stand-alone block
                    return built_block_node
                else
                    # Block should be part of a message send
                    built_send_node = translate_node(send_node)
                    built_send_node.block_argument = built_block_node
                    return built_send_node
                end
            end

            def translate_args(node)
                return translate_node(node.children)
            end

            def translate_arg(node)
                return node.children.first
            end
        end
    end
end