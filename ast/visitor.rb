require_relative "nodes.rb"

module Ikra
    module AST
        class Node
            def accept(visitor)
                visitor.visit_node(self)
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
        
        class IntNode
            def accept(visitor)
                visitor.visit_int_node(self)
            end
        end
        
        class FloatNode
            def accept(visitor)
                visitor.visit_float_node(self)
            end
        end
        
        class BoolNode
            def accept(visitor)
                visitor.visit_bool_node(self)
            end
        end
        
        class ForNode
            def accept(visitor)
                visitor.visit_for_node(self)
            end
        end
        
        class BreakNode
            def accept(visitor)
             visitor.visit_break_node(self)
            end
        end
        
        class IfNode
            def accept(visitor)
                visitor.visit_if_node(self)
            end
        end
        
        class BeginNode
            def accept(visitor)
                visitor.visit_begin_node(self)
            end
        end
        
        class SendNode
            def accept(visitor)
                visitor.visit_send_node(self)
            end
        end
        
        class Visitor
            def visit_node(node)
            
            end
            
            def visit_lvar_read_node(node)
            
            end
            
            def visit_lvar_write_node(node)
                node.value.accept(self)
            end
            
            def visit_int_node(node)
            
            end
            
            def visit_float_node(node)
            
            end
            
            def visit_bool_node(node)
            
            end
            
            def visit_for_node(node)
                node.range_from.accept(self)
                node.range_to.accept(self)
                node.body_stmts.accept(self)
            end
            
            def visit_break_node(node)
            
            end
            
            def visit_if_node(node)
                node.condition.accept(self)
                node.true_body_stmts.accept(self)
                node.false_body_stmts.accept(self)
            end
            
            def visit_begin_node(node)
                node.each do |stmt|
                    stmt.accept(self)
                end
            end
            
            def visit_send_node(node)
                # Receiver might be nil for self sends
                if node.receiver != nil
                    node.receiver.accept(self)
                end
                
                node.arguments.each do |arg|
                    arg.accept(self)
                end
            end
        end
    end
end