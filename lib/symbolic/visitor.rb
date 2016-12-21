module Ikra
    module Symbolic
        class ArrayIndexCommand
            def accept(visitor)
                visitor.visit_array_index_command(self)
            end
        end

        class ArrayCombineCommand
            def accept(visitor)
                visitor.visit_array_combine_command(self)
            end
        end

        class ArrayReduceCommand
            def accept(visitor)
                visitor.visit_array_reduce_command(self)
            end
        end

        class ArrayStencilCommand
            def accept(visitor)
                visitor.visit_array_stencil_command(self)
            end
        end

        class ArraySelectCommand
            def accept(visitor)
                visitor.visit_array_select_command(self)
            end
        end

        class ArrayIdentityCommand
            def accept(visitor)
                visitor.visit_array_identity_command(self)
            end
        end

        class ArrayZipCommand
            def accept(visitor)
                visitor.visit_array_zip_command(self)
            end
        end

        class Visitor
            def visit_array_command(command)
                for input in command.input
                    if input.command.is_a?(ArrayCommand)
                        input.command.accept(self)
                    end
                end
            end

            def visit_array_index_command(command)
                visit_array_command(command)
            end

            def visit_array_combine_command(command)
                visit_array_command(command)
            end

            def visit_array_reduce_command(command)
                visit_array_command(command)
            end

            def visit_array_stencil_command(command)
                visit_array_command(command)
            end

            def visit_array_select_command(command)
                visit_array_command(command)
            end

            def visit_array_identity_command(command)
                visit_array_command(command)
            end

            def visit_array_zip_command(command)
                visit_array_command(command)
            end
        end
    end
end