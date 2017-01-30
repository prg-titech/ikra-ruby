module Ikra
    module AST
        class Node
            def to_s
                return "[Node]"
            end
        end

        class ProgramNode
            def to_s
                return "[ProgramNode: #{blocks.to_s}; #{classes.to_s}]"
            end
        end

        class ClassDefNode
            def to_s
                return "[ClassDefNode: #{name}; #{instance_variables.to_s}; #{instance_methods.to_s}]"
            end
        end

        class VarDefNode
            def to_s
                return "[VarDefNode: #{name}, read = #{read}, written = #{written}]"
            end
        end

        class MethDefNode
            def to_s
                return "[MethDefNode: #{name} #{body.to_s}]"
            end
        end

        class BlockDefNode
            def to_s
                return "[BlockDefNode: #{body.to_s}]"
            end
        end

        class RootNode
            def to_s
                return "[RootNode: #{single_child.to_s}]"
            end
        end

        class SourceCodeExprNode
            def to_s
                return "[SourceCodeExprNode: #{code}]"
            end
        end
        
        class ConstNode
            def to_s
                return "[ConstNode: #{identifier.to_s}]"
            end
        end

        class LVarReadNode
            def to_s
                return "[LVarReadNode: #{identifier.to_s}]"
            end
        end

        class LVarWriteNode
            def to_s
                return "[LVarWriteNode: #{identifier.to_s} := #{value.to_s}]"
            end
        end

        class IntLiteralNode
            def to_s
                return "<#{value.to_s}>"
            end
        end

        class FloatLiteralNode
            def to_s
                return "<#{value.to_s}>"
            end
        end

        class BoolLiteralNode
            def to_s
                return "<#{value.to_s}>"
            end
        end

        class NilLiteralNode
            def to_s
                return "<nil>"
            end
        end

        class ForNode
            def to_s
                return "[ForNode: #{iterator_identifier.to_s} := #{range_from.to_s}...#{range_to.to_s}, #{body_stmts.to_s}]"
            end
        end

        class WhileNode
            def to_s
                return "[WhileNode: #{condition.to_s}, #{body_stmts.to_s}]"
            end
        end

        class WhilePostNode
            def to_s
                return "[WhilePostNode: #{condition.to_s}, #{body_stmts.to_s}]"
            end
        end

        class UntilNode
            def to_s
                return "[UntilNode: #{condition.to_s}, #{body_stmts.to_s}]"
            end
        end

        class UntilPostNode
            def to_s
                return "[UntilPostNode: #{condition.to_s}, #{body_stmts.to_s}]"
            end
        end

        class BreakNode
            def to_s
                return "[BreakNode]"
            end
        end

        class IfNode
            def to_s
                if false_body_stmts != nil
                    return "[IfNode: #{condition.to_s}, #{true_body_stmts.to_s}, #{false_body_stmts.to_s}]"
                else
                    return "[IfNode: #{condition.to_s}, #{true_body_stmts.to_s}]"
                end
            end
        end

        class BeginNode
            def to_s
                stmts = body_stmts.map do |stmt|
                    stmt.to_s
                end.join(";\n")

                return "[BeginNode: {#{stmts}}]"
            end
        end

        class SendNode
            def to_s
                args = arguments.map do |arg|
                    arg.to_s
                end.join("; ")

                return "[SendNode: #{receiver.to_s}.#{selector.to_s}(#{args})]"
            end
        end

        class ReturnNode
            def to_s
                return "[ReturnNode: #{value.to_s}]"
            end
        end
    end
end