require_relative "../ast/nodes"
require_relative "../ast/visitor"

module Ikra
    module Translator
        # Visitor that replaces implicit returns with explicit ones
        class LastStatementReturnsVisitor < AST::Visitor
            def visit_root_node(node)
                node.single_child.accept(self)
            end

            def visit_lvar_read_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end
            
            def visit_lvar_write_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end
            
            def visit_int_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end
            
            def visit_float_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end
            
            def visit_bool_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end
            
            def visit_for_node(node)
                raise "Cannot handle for loop as return value"
            end
            
            def visit_break_node(node)
                raise "Break must not be a return value"
            end
            
            def visit_if_node(node)
                node.true_body_stmts.accept(self)
                node.false_body_stmts.accept(self)
            end
            
            def visit_begin_node(node)
                node.body_stmts.last.accept(self)
            end
            
            def visit_send_node(node)
                node.parent.replace_child(node, AST::ReturnNode.new(value: node))
            end

            def visit_return_node(node)
                # Do nothing
            end
        end
    end
end