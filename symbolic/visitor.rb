module Ikra
    module Symbolic
        class ArrayNewCommand
            def accept(visitor)
                visitor.visit_array_new_command(self)
            end
        end

        class ArrayMapCommand
            def accept(visitor)
                visitor.visit_array_map_command(self)
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

        class Visitor
            def visit_array_command(command)

            end

            def visit_array_new_command(command)
                visit_array_command(command)
            end

            def visit_array_map_command(command)
                visit_array_command(command)
                command.target.accept(self)
            end

            def visit_array_select_command(command)
                visit_array_command(command)
                command.target.accept(self)
            end

            def visit_array_identity_command(command)
                visit_array_command(command)
            end
        end
    end
end