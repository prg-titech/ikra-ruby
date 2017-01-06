module Ikra
    module Symbolic
        class IterativeCommandWrapper
            def result_variable
                if base?
                    return "iterative_result_#{command.unique_id}"
                else
                    raise "Non-base command wrapper does not have a result identifier"
                end
            end
        end
    end

    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_iterative_command_wrapper(wrapper)
                # TODO: what to return?
            end

            def visit_iterative_computation(computation)

            end

            # Entry point into translation of iterative computations, unless we translate an
            # entire iterative computation directly by calling `execute`.
            def visit_iterative_computation_result_command(command)

            end
        end
    end
end
