require_relative "nodes"

module Ikra
    module AST
        module Builder
            class << self
                def translate_ast(node)
                    if node == nil
                        nil
                    else
                        send("translate_#{node.type.to_s}".to_sym, node)
                    end
                end
                
                def translate_int(node)
                    IntNode.new(value: node.children[0])
                end
                
                def translate_float(node)
                    FloatNode.new(value: node.children[0])
                end
                
                def translate_bool(node)
                    BoolNode.new(value: node.children[0])
                end
                
                def translate_lvar(node)
                    LVarReadNode.new(identifier: node.children[0])
                end
                
                def translate_lvasgn(node)
                    LVarWriteNode.new(identifier: node.children[0], value: translate_ast(node.children[1]))
                end
                
                def translate_if(node)
                    IfNode.new(condition: translate_ast(node.children[0]),
                        true_body_stmts: translate_ast(node.children[1]),
                        false_body_stmts: translate_ast(node.children[2]))
                end
                
                def extract_begin_single_statement(node, should_return = false)
                    next_node = node
                    while next_node.type == :begin
                        if next_node.children.size != 1
                            raise "Begin node contains more than one statement"
                        end
                        
                        next_node = next_node.children[0]
                    end
                    
                    next_node
                end
                
                def translate_for(node)
                    if node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :irange
                        range = extract_begin_single_statement(node.children[1])
                        
                        ForNode.new(iterator_identifier: node.children[0].children[0],
                            range_from: translate_ast(range.children[0]),
                            range_to: translate_ast(range.children[1]),
                            body_stmts: translate_ast(node.children[2]))
                    else
                        raise "Can only handle simple For loops at the moment"
                    end
                end
                
                def translate_break(node)
                    BreakNode.new
                end
                
                def translate_send(node)
                    SendNode.new(receiver: translate_ast(node.children[0]),
                        selector: node.children[1],
                        arguments: node.children[2..-1].map do |arg|
                            translate_ast(arg) end)
                end
                
                def translate_begin(node)
                    BeginNode.new(body_stmts: node.children.map do |stmt|
                        translate_ast(stmt) end)
                end
            end
        end
    end
end