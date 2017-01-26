require_relative "../../ast/nodes"
require_relative "../../ast/visitor"

module Ikra
    module Translator

        # This visitor finds parallel sections (send nodes), generates the source code for these
        # parallel sections, and replaces the send nodes with nodes that first execute the section
        # and then return the result.
        class ParallelSectionTranslator < AST::Visitor
            def initialize(command_translator:)
                @command_translator = command_translator
            end

            def visit_send_node(node)
                receiver_type = node.receiver.get_type

                if receiver_type.is_singleton? && 
                        receiver_type.singleton_type.is_a?(Symbolic::ArrayCommand)

                    # The result type is the symbolically executed result of applying this
                    # parallel section. The result type is an ArrayCommand.
                    array_command = node.get_type.singleton_type

                    # Translate command
                    @command_translator.push_kernel_launcher
                    result = array_command.accept(@command_translator)
                    kernel_launcher = @command_translator.pop_kernel_launcher(result)

                    node.parent.replace_child(node, AST::KernelLauncherNode.new(
                        kernel_launcher: kernel_launcher))
                end
            end
        end
    end

    module AST
        class KernelLauncherNode < Node
            attr_reader :kernel_launcher

            def initialize(kernel_launcher:)
                @kernel_launcher = kernel_launcher
            end

            def translate_expression
                launch_code = @kernel_launcher.build_kernel_launcher

                # Always return a device pointer. Only at the very end, we transfer data to
                # the host.
                result_expr = @kernel_launcher.kernel_result_var_name

                return "({ 
    #{launch_code} 
    #{Translator::Constants::PROGRAM_RESULT_IDENTIFIER}->result = #{result_expr}; 
    #{Translator::Constants::PROGRAM_RESULT_IDENTIFIER}; })"
            end
        end
    end
end