module Ikra
    module AST
        class Node
            attr_accessor :parent

            def is_begin_node?
                false
            end

            def replace(another_node)
                parent.replace_child(self, another_node)
            end

            def replace_child(node, another_node)
                instance_variables do |inst_var|
                    if instance_variable_get(inst_var) == node
                        instance_variable_set(inst_var, another_node)
                        another_node.parent = self
                    end
                end
            end
        end
        
        class LVarReadNode < Node
            attr_reader :identifier
            
            def initialize(identifier:)
                @identifier = identifier
            end
        end
        
        class LVarWriteNode < Node
            attr_reader :identifier
            attr_reader :value
            
            def initialize(identifier:, value:)
                @identifier = identifier
                @value = value

                value.parent = self
            end
        end
        
        class IntNode < Node
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end
        end
        
        class FloatNode < Node
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end
        end
        
        class BoolNode < Node
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end
        end
        
        class ForNode < Node
            attr_reader :iterator_identifier
            attr_reader :range_from
            attr_reader :range_to
            attr_reader :body_stmts
            
            def initialize(iterator_identifier:, range_from:, range_to:, body_stmts: BeginNode.new)
                @iterator_identifier = iterator_identifier
                @range_from = range_from
                @range_to = range_to
                @body_stmts = body_stmts

                range_from.parent = self
                range_to.parent = self
                body_stmts.parent = self
            end
        end
        
        class BreakNode < Node
        
        end
        
        class IfNode < Node
            attr_reader :condition
            attr_reader :true_body_stmts
            attr_reader :false_body_stmts
            
            def initialize(condition:, true_body_stmts:, false_body_stmts: nil)
                @condition = condition
                @true_body_stmts = true_body_stmts
                @false_body_stmts = false_body_stmts

                condition.parent = self
                true_body_stmts.parent = self

                if false_body_stmts != nil
                    false_body_stmts.parent = self
                end
            end
        end
        
        class BeginNode < Node
            attr_reader :body_stmts
            
            def initialize(body_stmts: [])
                @body_stmts = body_stmts

                body_stmts.each do |stmt|
                    stmt.parent = self
                end
            end
            
            def replace_child(node, another_node)
                @body_stmts = @body_stmts.map do |stmt|
                    if node == stmt
                        another_node.parent = self
                        another_node
                    else
                        stmt
                    end
                end
            end

            def is_begin_node?
                true
            end
        end
        
        class SendNode < Node
            attr_reader :receiver
            attr_reader :selector
            attr_reader :arguments
            
            def initialize(receiver:, selector:, arguments: [])
                @receiver = receiver
                @selector = selector
                @arguments = arguments

                receiver.parent = self
                arguments.each do |arg|
                    arg.parent = self
                end
            end

            def replace_child(node, another_node)
                if @receiver == node
                    @receiver = another_node
                end

                @arguments = @arguments.map do |arg|
                    if node == arg
                        another_node.parent = self
                        another_node
                    else
                        arg
                    end
                end
            end
        end

        class ReturnNode < Node
            attr_reader :value

            def initialize(value:)
                @value = value

                value.parent = self
            end
        end
    end
end