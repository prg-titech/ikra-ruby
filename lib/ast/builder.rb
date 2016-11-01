require_relative "nodes"

module Ikra
    module AST

        # Builds an Ikra (Ruby) AST from a Parser RubyGem AST
        module Builder
            class << self
                def from_parser_ast(node)
                    RootNode.new(single_child: translate_node(node))
                end

                private

                def translate_node(node)
                    if node == nil
                        nil
                    else
                        send("translate_#{node.type.to_s}".to_sym, node)
                    end
                end
                
                def translate_const(node)
                    # TODO(matthias): what is the meaning of the first child?
                    ConstNode.new(identifier: node.children[1])
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
                
                def translate_true(node)
                    BoolNode.new(value: true)
                end
                
                def translate_false(node)
                    BoolNode.new(value: false)
                end
                
                def translate_and(node)
                    SendNode.new(receiver: translate_node(node.children[0]),
                        selector: :"&&",
                        arguments: [translate_node(node.children[1])])
                end
                
                def translate_or(node)
                    SendNode.new(receiver: translate_node(node.children[0]),
                        selector: :"||",
                        arguments: [translate_node(node.children[1])])
                end
                
                def translate_lvar(node)
                    LVarReadNode.new(identifier: node.children[0])
                end
                
                def translate_lvasgn(node)
                    LVarWriteNode.new(identifier: node.children[0], value: translate_node(node.children[1]))
                end
                
                def translate_ivar(node)
                    IVarReadNode.new(identifier: node.children[0])
                end

                def translate_if(node)
                    IfNode.new(condition: translate_node(node.children[0]),
                        true_body_stmts: translate_node(node.children[1]),
                        false_body_stmts: translate_node(node.children[2]))
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
                            range_from: translate_node(range.children[0]),
                            range_to: translate_node(range.children[1]),
                            body_stmts: translate_node(node.children[2]))
                    elsif node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :erange
                        # Convert exclusive range to inclusive range
                        range = extract_begin_single_statement(node.children[1])
                        range_to = translate_node(range.children[1])
                        range_to_inclusive = SendNode.new(
                            receiver: range_to,
                            selector: :-,
                            arguments: [IntNode.new(value: 1)])

                        ForNode.new(iterator_identifier: node.children[0].children[0],
                            range_from: translate_node(range.children[0]),
                            range_to: range_to_inclusive,
                            body_stmts: translate_node(node.children[2]))
                    else
                        raise "Can only handle simple For loops at the moment"
                    end
                end
                
                def translate_while(node)
                    WhileNode.new(
                        condition: translate_node(node.children[0]),
                        body_stmts: translate_node(node.children[1]))
                end

                def translate_break(node)
                    BreakNode.new
                end
                
                def translate_return(node)
                    ReturnNode.new(value: translate_node(node.children[0]))
                end

                def translate_send(node)
                    receiver = nil

                    if node.children[0] == nil
                        # Implicit receiver
                        # TODO: Assuming Object for now, but this is not always correct
                        receiver = ConstNode.new(identifier: :Object)
                    else
                        receiver = translate_node(node.children[0])
                    end

                    SendNode.new(receiver: receiver,
                        selector: node.children[1],
                        arguments: node.children[2..-1].map do |arg|
                            translate_node(arg) end)
                end
                
                def translate_begin(node)
                    BeginNode.new(body_stmts: node.children.map do |stmt|
                        translate_node(stmt) end)
                end
            end
        end
    end
end
