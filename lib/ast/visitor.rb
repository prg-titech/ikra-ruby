require_relative "nodes.rb"

module Ikra
    module AST
        class Node
            def accept(visitor)
                return visitor.visit_node(self)
            end
        end
        
        class ProgramNode
            def accept(visitor)
                def accept(visitor)
                    return visitor.visit_program_node(self)
                end
            end
        end

        class ClassDefNode
            def accept(visitor)
                return visitor.visit_class_def_node(self)
            end
        end

        class VarDefNode
            def accept(visitor)
                return visitor.visit_var_def_node(self)
            end
        end

        class MethDefNode
            def accept(visitor)
                return visitor.visit_meth_def_node(self)
            end
        end

        class BlockDefNode
            def accept(visitor)
                return visitor.visit_block_def_node(self)
            end
        end

        class RootNode
            def accept(visitor)
                visitor.visit_root_node(self)
            end
        end

        class ArrayNode
            def accept(visitor)
                visitor.visit_array_node(self)
            end
        end

        class SourceCodeExprNode
            def accept(visitor)
                visitor.visit_source_code_expr_node(self)
            end
        end

        class HashNode
            def accept(visitor)
                visitor.visit_hash_node(self)
            end
        end

        class ConstNode
            def accept(visitor)
                visitor.visit_const_node(self)
            end
        end

        class LVarReadNode
            def accept(visitor)
                visitor.visit_lvar_read_node(self)
            end
        end
        
        class LVarWriteNode
            def accept(visitor)
                visitor.visit_lvar_write_node(self)
            end
        end
        
        class IVarReadNode
            def accept(visitor)
                visitor.visit_ivar_read_node(self)
            end
        end

        class IntLiteralNode
            def accept(visitor)
                visitor.visit_int_node(self)
            end
        end
        
        class FloatLiteralNode
            def accept(visitor)
                visitor.visit_float_node(self)
            end
        end
        
        class BoolLiteralNode
            def accept(visitor)
                visitor.visit_bool_node(self)
            end
        end
        
        class NilLiteralNode
            def accept(visitor)
                visitor.visit_nil_node(self)
            end
        end

        class SymbolLiteralNode
            def accept(visitor)
                visitor.visit_symbol_node(self)
            end
        end

        class StringLiteralNode
            def accept(visitor)
                visitor.visit_string_node(self)
            end
        end
        
        class ForNode
            def accept(visitor)
                visitor.visit_for_node(self)
            end
        end
        
        class WhileNode
            def accept(visitor)
                return visitor.visit_while_node(self)
            end
        end
        
        class WhilePostNode
            def accept(visitor)
                return visitor.visit_while_post_node(self)
            end
        end
        
        class UntilNode
            def accept(visitor)
                return visitor.visit_until_node(self)
            end
        end
        
        class UntilPostNode
            def accept(visitor)
                return visitor.visit_until_post_node(self)
            end
        end

        class BreakNode
            def accept(visitor)
             return visitor.visit_break_node(self)
            end
        end
        
        class IfNode
            def accept(visitor)
                return visitor.visit_if_node(self)
            end
        end
        
        class TernaryNode
            def accept(visitor)
                return visitor.visit_ternary_node(self)
            end
        end
        
        class BeginNode
            def accept(visitor)
                return visitor.visit_begin_node(self)
            end
        end
        
        class SendNode
            def accept(visitor)
                return visitor.visit_send_node(self)
            end
        end
        
        class ReturnNode
            def accept(visitor)
                return visitor.visit_return_node(self)
            end
        end

        class Visitor
            def visit_node(node)
            
            end

            def visit_program_node(node)
                visit_node(node)

                node.classes.each do |c|
                    c.accept(self)
                end

                node.blocks.each do |b|
                    b.accept(self)
                end
            end

            def visit_class_def_node(node)
                visit_node(node)

                node.instance_variables.each do |iv|
                    iv.accept(self)
                end

                node.instance_methods.each do |im|
                    im.accept(self)
                end
            end

            def visit_root_node(node)
                visit_node(node)
                node.single_child.accept(self)
            end

            def visit_array_node(node)
                visit_node(node)

                for value in node.values
                    value.accept(self)
                end
            end

            def visit_source_code_expr_node(node)
                visit_node(node)
            end

            def visit_hash_node(node)
                visit_node(node)

                node.hash.each do |key, value|
                    key.accept(self)
                    value.accept(self)
                end
            end

            def visit_var_def_node(node)
                visit_node(node)
            end

            def visit_meth_def_node(node)
                visit_node(node)
                node.body.accept(self)
            end

            def visit_block_def_node(node)
                visit_node(node)
                node.body.accept(self)
            end

            def visit_const_node(node)
                visit_node(node)
            end
            
            def visit_lvar_read_node(node)
                visit_node(node)
            end
            
            def visit_lvar_write_node(node)
                visit_node(node)
                node.value.accept(self)
            end
            
            def visit_ivar_read_node(node)
                visit_node(node)
            end
            
            def visit_int_node(node)
                visit_node(node)
            end
            
            def visit_float_node(node)
                visit_node(node)
            
            end
            
            def visit_bool_node(node)
                visit_node(node)
            end
            
            def visit_nil_node(node)
                visit_node(node)
            end
            
            def visit_symbol_node(node)
                visit_node(node)
            end

            def visit_string_node(node)
                visit_node(node)
            end

            def visit_for_node(node)
                visit_node(node)

                node.range_from.accept(self)
                node.range_to.accept(self)
                node.body_stmts.accept(self)
            end
            
            def visit_while_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.body_stmts.accept(self)
            end
            
            def visit_while_post_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.body_stmts.accept(self)
            end
            
            def visit_until_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.body_stmts.accept(self)
            end
            
            def visit_until_post_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.body_stmts.accept(self)
            end

            def visit_break_node(node)
                visit_node(node)
            end
            
            def visit_if_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.true_body_stmts.accept(self)
                node.false_body_stmts.accept(self)
            end
            
            def visit_ternary_node(node)
                visit_node(node)

                node.condition.accept(self)
                node.true_val.accept(self)
                node.false_val.accept(self)
            end
            
            def visit_begin_node(node)
                visit_node(node)

                node.body_stmts.each do |stmt|
                    stmt.accept(self)
                end
            end
            
            def visit_send_node(node)
                visit_node(node)

                # Receiver might be nil for self sends
                if node.receiver != nil
                    node.receiver.accept(self)
                end
                
                node.arguments.each do |arg|
                    arg.accept(self)
                end
            end

            def visit_return_node(node)
                visit_node(node)
                node.value.accept(self)
            end
        end
    end
end