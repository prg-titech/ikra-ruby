module Ikra
    module AST
        class Node
            attr_accessor :parent

            def eql?(other)
                return self == other
            end

            def ==(other)
                return self.class == other.class
            end

            def hash
                return 1231
            end
        end

        class ProgramNode < Node
            # First block is program entry point
            attr_reader :blocks
            attr_reader :classes

            def initialize(blocks: [], classes: [])
                @blocks = blocks
                @classes = classes
            end

            def clone
                return ProgramNode.new(
                    blocks: @blocks.map do |b| b.clone end,
                    classes: @classes.map do |c| c.clone end)
            end

            def ==(other)
                return super(other) && blocks == other.blocks && classes == other.classes
            end

            def hash
                return (blocks.hash + classes.hash) % 4524321
            end
        end

        class ClassDefNode < Node
            attr_reader :name
            attr_reader :instance_variables
            attr_reader :instance_methods
            attr_reader :ruby_class
            
            # Class variables/methods are defined as instance variables/methods on the singleton
            # class ClassDefNode

            def initialize(
                    name:, 
                    ruby_class:, 
                    instance_variables: [], 
                    instance_methods: [], 
                    class_variables: [], 
                    class_methods: [])
                @name = name
                @ruby_class = ruby_class
                @instance_variables = instance_variables
                @instance_methods = instance_methods
            end

            def clone
                return ClassDefNode.new(
                    name: @name,
                    ruby_class: @ruby_class,
                    instance_variables: @instance_variables.map do |i| i.clone end,
                    instance_methods: @instance_methods.map do |i| i.clone end,
                    class_variables: @class_variables.map do |c| c.clone end,
                    class_methods: @class_methods.map do |c| c.clone end)
            end

            def add_instance_variable(inst_var)
                instance_variables.push(inst_var)
                inst_meth.parent = self
            end

            def add_instance_method(inst_meth)
                instance_methods.push(inst_meth)
                inst_meth.parent = self
            end

            def has_instance_method?(selector)
                return instance_method(selector) != nil
            end

            def instance_method(selector)
                return instance_methods.find do |meth|
                    meth.name == selector
                end
            end

            def enclosing_class
                return self
            end

            def ==(other)
                return super(other) && 
                    name == other.name &&
                    ruby_class == other.ruby_class &&
                    instance_variables == other.instance_variables &&
                    instance_methods == other.instance_methods &&
                    class_variables == other.class_variables &&
                    class_methods == other.class_methods
            end
        end

        class VarDefNode < Node
            attr_reader :name
            attr_accessor :read
            attr_accessor :written

            def initialize(name:, read: false, written: false)
                @name = name
                @read = read
                @written = written
            end

            def clone
                return VarDefNode.new(
                    name: @name,
                    read: @read,
                    written: @written)
            end

            def ==(other)
                return super(other) &&
                    name == other.name &&
                    read == other.read &&
                    written == other.written
            end
        end

        class BehaviorNode < Node
            def find_behavior_node
                return self
            end
        end

        class MethDefNode < BehaviorNode
            attr_reader :name
            attr_reader :ruby_method
            attr_reader :body

            def initialize(name:, body:, ruby_method:, method_binding: nil)
                @name = name
                @body = body
                @ruby_method = ruby_method
                @binding = method_binding

                body.parent = self
            end

            def clone
                return MethodDefNode.new(
                    name: @name,
                    body: @body.clone,
                    ruby_method: @ruby_method)
            end

            def binding
                if @binding != nil
                    return @binding
                elsif ruby_method == nil 
                    return nil
                else 
                    return ruby_method.send(:binding)
                end
            end

            def ==(other)
                return super(other) && name == other.name && body == other.body
            end
        end

        class BlockDefNode < BehaviorNode
            attr_reader :body
            attr_reader :ruby_block
            attr_reader :parameters

            def initialize(body:, ruby_block:, parameters: nil)
                @body = body
                @ruby_block = ruby_block
                @parameters = parameters

                body.parent = self
            end

            def clone
                return BlockDefNode.new(
                    body: @body.clone,
                    ruby_block: @ruby_block,
                    parameters: @parameters == nil ? nil : @parameters.dup)
            end

            def binding
                return ruby_block.binding
            end

            def ==(other)
                return super(other) && body == other.body && parameters == other.parameters
            end
        end

        class TreeNode < Node
            def is_begin_node?
                false
            end

            def replace(another_node)
                # Sometimes, this method does not work as expected, if the `parent` of the `self`
                # is already modified before calling this method.
                parent.replace_child(self, another_node)
            end

            def replace_child(node, another_node)
                instance_variables.each do |inst_var|
                    if instance_variable_get(inst_var).equal?(node)
                        instance_variable_set(inst_var, another_node)
                        another_node.parent = self
                    end
                end
            end

            def enclosing_class
                @parent.enclosing_class
            end

            def find_behavior_node
                return parent.find_behavior_node
            end

            TYPE_INFO_VARS = [:@return_type_by_recv_type, :@type]

            def ==(other)
                if self.class != other.class
                    return false
                end

                # Ignore types
                if (instance_variables - TYPE_INFO_VARS) != (other.instance_variables - TYPE_INFO_VARS)
                    
                    return false
                end

                for var_name in instance_variables
                    if var_name != :@parent && !TYPE_INFO_VARS.include?(var_name)
                        # Avoid cycles via :parent... There could still be other cycles though
                        if instance_variable_get(var_name) != other.instance_variable_get(var_name)
                            return false
                        end
                    end
                end

                return true
            end
        end

        # Need to wrap block bodies in RootNode, so that the first node can be replaced if necessary (LastStatementReturnsVisitor)
        class RootNode < TreeNode
            attr_reader :single_child

            def initialize(single_child:)
                @single_child = single_child
                single_child.parent = self
            end

            def clone
                return RootNode.new(single_child: @single_child.clone)
            end
        end

        class ArrayNode < TreeNode
            attr_reader :values

            def initialize(values:)
                @values = values

                for value in values
                    value.parent = self
                end
            end

            def clone
                return ArrayNode.new(
                    values: @values.map do |v| v.clone end)
            end
        end

        # A synthetic AST node. Contains its string translation directly.
        class SourceCodeExprNode < TreeNode
            attr_reader :code

            def initialize(code:)
                @code = code
            end

            def clone
                return SourceCodeExprNode.new(code: @code)
            end
        end

        class HashNode < TreeNode
            attr_reader :hash

            def initialize(hash:)
                @hash = hash
            end

            def clone
                # TODO: Clone properly
                return HashNode.new(hash: @hash.clone)
            end
        end

        class ConstNode < TreeNode
            attr_reader :identifier

            def initialize(identifier:)
                @identifier = identifier
            end

            def clone
                return ConstNode.new(identifier: @identifier)
            end
        end

        class LVarReadNode < TreeNode
            attr_reader :identifier
            
            def initialize(identifier:)
                @identifier = identifier
            end

            def clone
                return LVarReadNode.new(identifier: @identifier)
            end
        end
        
        class LVarWriteNode < TreeNode
            attr_reader :identifier
            attr_reader :value
            
            def initialize(identifier:, value:)
                @identifier = identifier
                @value = value

                value.parent = self
            end

            def clone
                return LVarWriteNode.new(
                    identifier: @identifier,
                    value: @value.clone)
            end
        end
        
        class IVarReadNode < TreeNode
            attr_reader :identifier

            def initialize(identifier:)
                @identifier = identifier
            end

            def clone
                return IVarReadNode.new(identifier: @identifier)
            end
        end

        class IntLiteralNode < TreeNode
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end

            def clone
                return IntLiteralNode.new(value: @value)
            end
        end
        
        class FloatLiteralNode < TreeNode
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end

            def clone
                return FloatLiteralNode.new(value: @value)
            end
        end
        
        class BoolLiteralNode < TreeNode
            attr_reader :value
            
            def initialize(value:)
                @value = value
            end

            def clone
                return BoolLiteralNode.new(value: @value)
            end
        end

        class NilLiteralNode < TreeNode
            
        end
        
        class SymbolLiteralNode < TreeNode
            attr_reader :value

            def initialize(value:)
                @value = value
            end

            def clone
                return SymbolLiteralNode.new(value: @value)
            end
        end

        class StringLiteralNode < TreeNode
            attr_reader :value

            def initialize(value:)
                @value = value
            end

            def clone
                return StringLiteralNode.new(value: @value)
            end
        end

        class ForNode < TreeNode
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

            def clone
                return ForNode.new(
                    iterator_identifier: @iterator_identifier,
                    range_from: @range_from.clone,
                    range_to: @range_to.clone,
                    body_stmts: @body_stmts.clone)
            end
        end
        
        class WhileNode < TreeNode
            attr_reader :condition
            attr_reader :body_stmts

            def initialize(condition:, body_stmts:)
                @condition = condition
                @body_stmts = body_stmts

                condition.parent = self
                body_stmts.parent = self
            end

            def clone
                return WhileNode.new(
                    condition: @condition.clone,
                    body_stmts: @body_stmts.clone)
            end
        end
        
        class WhilePostNode < TreeNode
            attr_reader :condition
            attr_reader :body_stmts

            def initialize(condition:, body_stmts:)
                @condition = condition
                @body_stmts = body_stmts

                condition.parent = self
                body_stmts.parent = self
            end

            def clone
                return WhilePostNode.new(
                    condition: @condition.clone,
                    body_stmts: @body_stmts.clone)
            end
        end
        
        class UntilNode < TreeNode
            attr_reader :condition
            attr_reader :body_stmts

            def initialize(condition:, body_stmts:)
                @condition = condition
                @body_stmts = body_stmts

                condition.parent = self
                body_stmts.parent = self
            end

            def clone
                return UntilNode.new(
                    condition: @condition.clone,
                    body_stmts: @body_stmts.clone)
            end
        end
        
        class UntilPostNode < TreeNode
            attr_reader :condition
            attr_reader :body_stmts

            def initialize(condition:, body_stmts:)
                @condition = condition
                @body_stmts = body_stmts

                condition.parent = self
                body_stmts.parent = self
            end

            def clone
                return UntilPostNode.new(
                    condition: @condition.clone,
                    body_stmts: @body_stmts.clone)
            end
        end

        class BreakNode < TreeNode
        
        end
        
        class IfNode < TreeNode
            attr_reader :condition
            attr_reader :true_body_stmts
            attr_reader :false_body_stmts
            
            def initialize(condition:, true_body_stmts:, false_body_stmts: nil)
                if true_body_stmts == nil
                    # Handle empty `true` block
                    true_body_stmts = BeginNode.new
                end

                if false_body_stmts == nil
                    # Handle empty `false` block
                    false_body_stmts = BeginNode.new
                end

                @condition = condition
                @true_body_stmts = true_body_stmts
                @false_body_stmts = false_body_stmts

                condition.parent = self
                true_body_stmts.parent = self 
                false_body_stmts.parent = self
            end

            def clone
                return IfNode.new(
                    condition: @condition.clone,
                    true_body_stmts: @true_body_stmts.clone,
                    false_body_stmts: @false_body_stmts.clone)
            end
        end
        
        class TernaryNode < TreeNode
            attr_reader :condition
            attr_reader :true_val
            attr_reader :false_val
            
            def initialize(condition:, true_val:, false_val:)
                @condition = condition
                @true_val = true_val
                @false_val = false_val

                condition.parent = self
                true_val.parent = self
                false_val.parent = self
            end

            def clone
                return TernaryNode.new(
                    condition: @condition.clone,
                    true_val: @true_val.clone,
                    false_val: @false_val.clone)
            end
        end
        
        class BeginNode < TreeNode
            attr_reader :body_stmts
            
            def initialize(body_stmts: [])
                @body_stmts = body_stmts

                body_stmts.each do |stmt|
                    stmt.parent = self
                end
            end

            def clone
                return BeginNode.new(
                    body_stmts: @body_stmts.map do |s| s.clone end)
            end

            def add_statement(node)
                body_stmts.push(node)
                node.parent = self
            end
            
            def replace_child(node, another_node)
                @body_stmts = @body_stmts.map do |stmt|
                    if node.equal?(stmt)
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
        
        class SendNode < TreeNode
            attr_reader :receiver
            attr_reader :selector
            attr_reader :arguments
            attr_reader :block_argument
            
            def initialize(receiver:, selector:, arguments: [], block_argument: nil)
                @receiver = receiver
                @selector = selector
                @arguments = arguments
                @block_argument = block_argument

                receiver.parent = self
                arguments.each do |arg|
                    arg.parent = self
                end
            end

            def clone
                return SendNode.new(
                    receiver: @receiver.clone,
                    selector: @selector,
                    arguments: @arguments.map do |a| a.clone end,
                    block_argument: block_argument == nil ? nil : block_argument.clone)
            end

            def replace_child(node, another_node)
                if @receiver.equal?(node)
                    @receiver = another_node
                    another_node.parent = self
                end

                @arguments = @arguments.map do |arg|
                    if node.equal?(arg)
                        another_node.parent = self
                        another_node
                    else
                        arg
                    end
                end
            end

            # Setter required for [HostSectionBuilder]
            def block_argument=(value)
                @block_argument = value
            end
        end

        class ReturnNode < TreeNode
            attr_reader :value

            def initialize(value:)
                @value = value

                value.parent = self
            end

            def clone
                return ReturnNode.new(value: value.clone)
            end
        end
    end
end