require_relative "input"
require_relative "visitor"

module Ikra
    module Symbolic
        class Input
            def accept(visitor)
                visitor.visit_input(self, pattern: pattern)
            end
        end

        class SingleInput
            def accept(visitor)
                visitor.visit_single_input(self, pattern: pattern)
            end
        end

        class StencilArrayInput
            def accept(visitor)
                visitor.visit_stecil_array_input(self, pattern: pattern)
            end
        end

        class StencilSingleInput
            def accept(visitor)
                visitor.visit_stencil_single_input(self, pattern: pattern)
            end
        end

        class ReduceInput
            def accept(visitor)
                visitor.visit_reduce_input(self, pattern: pattern)
            end
        end

        class InputVisitor < Visitor
            def visit_input(input, pattern:)

            end

            def visit_single_input(input, pattern:)
                visit_input(input)
                input.command.accept(self)
            end

            def visit_stencil_array_input(input, pattern:)
                visit_input(input)
                input.command.accept(self)
            end

            def visit_stencil_single_input(input, pattern:)
                visit_input(input)
                input.command.accept(self)
            end

            def visit_reduce_input(input, pattern:)
                visit_input(input)
                input.command.accept(self)
            end

            def visit_array_command(command)
                for input in command.input
                    input.accept(self)
                end
            end
        end
    end
end