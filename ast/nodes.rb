module Ikra
    module AST
        class Node
            def is_begin_node?
                false
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
            end
        end
        
        class BreakNode < Node
        
        end
        
        class IfNode < Node
            attr_reader :condition
            attr_reader :true_body_stmts
            attr_reader :false_body_stmts
            
            def initialize(condition:, true_body_stmts:, false_body_stmts: BeginNode.new)
                @condition = condition
                @true_body_stmts = true_body_stmts
                @false_body_stmts = false_body_stmts
            end
        end
        
        class BeginNode < Node
            attr_reader :body_stmts
            
            def initialize(body_stmts: [])
                @body_stmts = body_stmts
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
            end
        end
    end
end