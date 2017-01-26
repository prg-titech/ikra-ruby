require_relative "nodes"

module Ikra
    module AST

        # Builds an Ikra (Ruby) AST from a Parser RubyGem AST
        class Builder
            def self.from_parser_ast(node)
                return self.new.from_parser_ast(node)
            end

            def from_parser_ast(node)
                return RootNode.new(single_child: translate_node(node))
            end

            protected

            def wrap_in_begin(translated_node)
                if translated_node != nil
                    return BeginNode.new(body_stmts: [translated_node])
                else
                    return nil
                end
            end

            def translate_node(node)
                if node == nil
                    return nil
                else
                    if node.is_a?(Array)
                        # An array of nodes
                        return node.map do |n|
                            send("translate_#{n.type.to_s.gsub("-", "_")}".to_sym, n)
                        end
                    else
                        # A single node
                        return send("translate_#{node.type.to_s.gsub("-", "_")}".to_sym, node)
                    end
                end
            end
            
            def translate_const(node)
                # TODO(matthias): what is the meaning of the first child?
                return ConstNode.new(identifier: node.children[1])
            end

            def translate_int(node)
                return IntLiteralNode.new(value: node.children[0])
            end
            
            def translate_float(node)
                return FloatLiteralNode.new(value: node.children[0])
            end
            
            def translate_bool(node)
                return BoolLiteralNode.new(value: node.children[0])
            end
            
            def translate_true(node)
                return BoolLiteralNode.new(value: true)
            end
            
            def translate_false(node)
                return BoolLiteralNode.new(value: false)
            end

            def translate_nil(node)
                return NilLiteralNode.new
            end
            
            def translate_and(node)
                return SendNode.new(receiver: translate_node(node.children[0]),
                    selector: :"&&",
                    arguments: [translate_node(node.children[1])])
            end
            
            def translate_or(node)
                return SendNode.new(receiver: translate_node(node.children[0]),
                    selector: :"||",
                    arguments: [translate_node(node.children[1])])
            end
            
            def translate_lvar(node)
                return LVarReadNode.new(identifier: node.children[0])
            end
            
            def translate_lvasgn(node)
                return LVarWriteNode.new(identifier: node.children[0], value: translate_node(node.children[1]))
            end
            
            def translate_ivar(node)
                return IVarReadNode.new(identifier: node.children[0])
            end

            def translate_if(node)
                return IfNode.new(condition: translate_node(node.children[0]),
                    true_body_stmts: wrap_in_begin(translate_node(node.children[1])),
                    false_body_stmts: wrap_in_begin(translate_node(node.children[2])))
            end
            
            def extract_begin_single_statement(node, should_return = false)
                next_node = node
                while next_node.type == :begin
                    if next_node.children.size != 1
                        raise "Begin node contains more than one statement"
                    end
                    
                    next_node = next_node.children[0]
                end
                
                return next_node
            end
            
            def translate_for(node)
                if node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :irange
                    range = extract_begin_single_statement(node.children[1])
                    
                    return ForNode.new(iterator_identifier: node.children[0].children[0],
                        range_from: translate_node(range.children[0]),
                        range_to: translate_node(range.children[1]),
                        body_stmts: wrap_in_begin(translate_node(node.children[2])))
                elsif node.children[0].type == :lvasgn and extract_begin_single_statement(node.children[1]).type == :erange
                    # Convert exclusive range to inclusive range
                    range = extract_begin_single_statement(node.children[1])
                    range_to = translate_node(range.children[1])
                    range_to_inclusive = SendNode.new(
                        receiver: range_to,
                        selector: :-,
                        arguments: [IntLiteralNode.new(value: 1)])

                    return ForNode.new(iterator_identifier: node.children[0].children[0],
                        range_from: translate_node(range.children[0]),
                        range_to: range_to_inclusive,
                        body_stmts: wrap_in_begin(translate_node(node.children[2])))
                else
                    raise "Can only handle simple For loops at the moment"
                end
            end
            
            def translate_while(node)
                return WhileNode.new(
                    condition: translate_node(node.children[0]),
                    body_stmts: wrap_in_begin(translate_node(node.children[1])))
            end
            
            def translate_while_post(node)
                return WhilePostNode.new(
                    condition: translate_node(node.children[0]),
                    body_stmts: wrap_in_begin(translate_node(node.children[1])))
            end
            
            def translate_until(node)
                return UntilNode.new(
                    condition: (SendNode.new(receiver: translate_node(node.children[0]),
                        selector: :^,
                        arguments: [BoolLiteralNode.new(value: true)])),
                    body_stmts: wrap_in_begin(translate_node(node.children[1])))
            end
            
            def translate_until_post(node)
                return UntilPostNode.new(
                    condition: (SendNode.new(receiver: translate_node(node.children[0]),
                        selector: :^,
                        arguments: [BoolLiteralNode.new(value: true)])),
                    body_stmts: wrap_in_begin(translate_node(node.children[1])))
            end

            def translate_break(node)
                return BreakNode.new
            end
            
            def translate_return(node)
                return ReturnNode.new(value: translate_node(node.children[0]))
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

                return SendNode.new(receiver: receiver,
                    selector: node.children[1],
                    arguments: node.children[2..-1].map do |arg|
                        translate_node(arg) end)
            end
            
            def translate_begin(node)
                return BeginNode.new(body_stmts: node.children.map do |stmt|
                    translate_node(stmt) end)
            end
            
            def translate_kwbegin(node)
                return BeginNode.new(body_stmts: node.children.map do |stmt|
                    translate_node(stmt) end)
            end
        end
    end
end
