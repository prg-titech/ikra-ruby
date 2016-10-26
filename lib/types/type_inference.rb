require "set"

require_relative "../ast/nodes.rb"
require_relative "../ast/visitor.rb"

require_relative "primitive_type"
require_relative "symbol_table"
require_relative "union_type"
require_relative "ruby_extension"

module Ikra
    module AST
        class Node
            def get_type
                return @type ||= Types::UnionType.new
            end
        end

        class ClassDefNode
            def get_type
                return ruby_class.to_ikra_type
            end
        end

        class LVarReadNode
            attr_accessor :variable_type
        end

        class LVarWriteNode
            attr_accessor :variable_type
        end

        class MethDefNode
            attr_accessor :receiver_type

            def initialize_types(receiver_type:, return_type: Types::UnionType.new)
                @receiver_type = receiver_type
                @type = return_type
            end

            def self.new_with_types(
                    name:,
                    body:,
                    parameters_names_and_types:,
                    ruby_method:,
                    receiver_type:,
                    return_type: Types::UnionType.new)

                instance = new(name: name, body: body, ruby_method: ruby_method)
                instance.initialize_types(receiver_type: receiver_type, return_type: return_type)
                instance.parameters_names_and_types = parameters_names_and_types
                return instance
            end

            def callers
                @callers ||= []
            end
        end

        class BehaviorNode
            # Mapping: parameter name -> UnionType
            def parameters_names_and_types
                @parameters_names_and_types ||= {}
            end

            def parameters_names_and_types=(value)
                @parameters_names_and_types = value
            end

            # Mapping: lexical variable name -> UnionType
            def lexical_variables_names_and_types
                @lexical_variables_names_and_types ||= {}
            end

            # Mapping: local variable name -> UnionType
            def local_variables_names_and_types
                @local_variables_names_and_types ||= {}
            end

            def local_variables_names_and_types=(value)
                @local_variables_names_and_types = value
            end

            def symbol_table
                @symbol_table ||= TypeInference::SymbolTable.new
            end
        end
    end
    
    module TypeInference
        class Visitor < AST::Visitor
            attr_reader :classes

            def initialize
                # Ikra type -> ClassDefNode
                @classes = {}
                @classes.default_proc = proc do |hash, key|
                    hash[key] = AST::ClassDefNode.new(
                        name: key.cls.name,
                        ruby_class: key.cls)
                end

                # Top of stack is the method that is currently processed
                @work_stack = []

                # Block/method definitions that must be processed
                @worklist = Set.new
            end

            def all_methods
                return classes.values.map do |class_|
                    class_.instance_methods
                end.flatten
            end

            def symbol_table
                current_method_or_block.symbol_table
            end

            def binding
                current_method_or_block.binding
            end

            def current_method_or_block
                @work_stack.last
            end

            # This is used as an entry point for the visitor
            def process_block(block_def_node)
                Log.info("Type inference: proceed into block(#{Types::UnionType.parameter_hash_to_s(block_def_node.parameters_names_and_types)})")
                @work_stack.push(block_def_node)
                body_ast = block_def_node.body

                # Variables that are not defined inside the block
                predefined_variables = []

                # Add parameters to symbol table (name -> type)
                block_def_node.parameters_names_and_types.each do |name, type|
                    symbol_table.declare_variable(name, type: type)
                    predefined_variables.push(name)
                end

                # Add lexical variables to symbol table (name -> type)
                block_def_node.lexical_variables_names_and_types.each do |name, type|
                    # Variable might be shadowed by parameter
                    symbol_table.ensure_variable_declared(name, type: type, kind: :lexical)
                    predefined_variables.push(name)
                end

                # Add return statements
                body_ast.accept(Translator::LastStatementReturnsVisitor.new)

                # Infer types
                body_ast.accept(self)

                # Get local variable definitons
                for variable_name in symbol_table.read_and_written_variables
                    if !predefined_variables.include?(variable_name)
                        block_def_node.local_variables_names_and_types[variable_name] =
                            symbol_table[variable_name]
                    end
                end

                return_value_type = symbol_table.return_type 
                Log.info("Type inference: block return type is #{return_value_type.to_s}")

                @work_stack.pop

                block_def_node.get_type.expand_return_type(return_value_type)
            end

            # This is used as an entry point for the visitor
            def process_method(method_def_node)
                Log.info("Type inference: proceed into method #{method_def_node.receiver_type}.#{method_def_node.name}(#{Types::UnionType.parameter_hash_to_s(method_def_node.parameters_names_and_types)})")

                @work_stack.push(method_def_node)
                body_ast = method_def_node.body

                # TODO: handle multiple receiver types
                recv_type = method_def_node.receiver_type

                # Variables that are not defined inside the method
                predefined_variables = []

                # Add parameters to symbol table (name -> type)
                method_def_node.parameters_names_and_types.each do |name, type|
                    symbol_table.declare_variable(name, type: type)
                    predefined_variables.push(name)
                end

                # Add lexical variables to symbol table (name -> type)
                method_def_node.lexical_variables_names_and_types.each do |name, type|
                    # Variable might be shadowed by parameter
                    symbol_table.ensure_variable_declared(name, type: type, kind: :lexical)
                    predefined_variables.push(name)
                end

                # Add return statements
                body_ast.accept(Translator::LastStatementReturnsVisitor.new)

                # Infer types
                body_ast.accept(self)

                # Get local variable definitons
                for variable_name in symbol_table.read_and_written_variables
                    if !predefined_variables.include?(variable_name)
                        method_def_node.local_variables_names_and_types[variable_name] =
                            symbol_table[variable_name]
                    end
                end
                
                return_value_type = symbol_table.return_type 
                Log.info("Type inference: method return type is #{return_value_type.to_s}")

                @work_stack.pop

                method_def_node.get_type.expand_return_type(return_value_type)
            end

            # This is not an actual Visitor method. It is called from visit_send_node.
            def visit_method_call(send_node)
                recv_type = send_node.receiver.get_type
                selector = send_node.selector
                return_type = Types::UnionType.new

                recv_type.types.each do |recv_singleton_type|
                    parameter_names = recv_singleton_type.method_parameters(selector)
                    arg_types = send_node.arguments.map do |arg| arg.get_type end
                    ast = recv_singleton_type.method_ast(selector)
                    method_visited_before = nil

                    if not @classes[recv_singleton_type].has_instance_method?(selector)
                        # This method was never visited before
                        method_def_node = AST::MethDefNode.new_with_types(
                            name: selector,
                            body: ast,
                            parameters_names_and_types: Hash[*parameter_names.zip(
                                Array.new(arg_types.size) do 
                                    Types::UnionType.new
                                end).flatten],
                            ruby_method: nil,
                            receiver_type: recv_singleton_type)
                        @classes[recv_singleton_type].add_instance_method(method_def_node)
                        method_visited_before = false
                    else
                        method_visited_before = true
                    end

                    method_def_node = @classes[recv_singleton_type].instance_method(selector)
                    # Method needs processing if any parameter is expanded (or method was never visited before)
                    needs_processing = !method_visited_before or parameter_names.map.with_index do |name, index|
                        method_def_node.parameters_names_and_types[name].expand(arg_types[index])   # returns true if expanded 
                    end.reduce(:|)

                    # Return value type from the last pass
                    # TODO: Have to make a copy here?
                    last_return_type = method_def_node.get_type
                    
                    if needs_processing
                        process_method(method_def_node)
                    end

                    if not last_return_type.include_all?(method_def_node.get_type)
                        # Return type was expanded during this pass, reprocess all callers (except for current method)
                        @worklist += (method_def_node.callers - [current_method_or_block])
                    end

                    method_def_node.callers.push(current_method_or_block)

                    # Return value of all visit methods should be the type
                    return_type.expand(method_def_node.get_type)
                end

                send_node.get_type.expand(return_type)
                return_type
            end

            def assert_singleton_type(union_type, expected_type)
                if union_type.singleton_type != expected_type
                    raise "Expected type #{expected_type} but found #{union_type.singleton_type}"
                end
            end

            def visit_const_node(node)
                if not binding
                    raise "Unable to resolve constants without Binding"
                end

                constant = binding.eval(node.identifier.to_s)
                constant_class = nil

                if constant.is_a?(Module)
                    constant_class = constant.singleton_class
                else
                    constant_class = constant.class
                end

                node.get_type.expand_return_type(Types::UnionType.new(constant_class.to_ikra_type))
            end

            def visit_root_node(node)
                node.get_type.expand_return_type(node.single_child.accept(self))
            end

            def visit_lvar_read_node(node)
                symbol_table.read!(node.identifier)

                # Extend type of variable
                return node.get_type.expand_return_type(symbol_table[node.identifier])
            end

            def visit_lvar_write_node(node)
                type = node.value.accept(self)

                # Declare/extend type in symbol table
                symbol_table.ensure_variable_declared(node.identifier, type: type)
                symbol_table.written!(node.identifier)

                # Extend type of variable
                return node.get_type.expand_return_type(type)
            end
            
            def visit_ivar_read_node(node)
                cls_type = node.enclosing_class.ruby_class.to_ikra_type
                cls_type.inst_var_read!(node.identifier)
                cls_type.inst_vars_types[node.identifier]
            end

            def visit_int_node(node)
                node.get_type.expand_return_type(Types::UnionType.create_int)
            end
            
            def visit_float_node(node)
                node.get_type.expand_return_type(Types::UnionType.create_float)
            end
            
            def visit_bool_node(node)
                node.get_type.expand_return_type(Types::UnionType.create_bool)
            end
            
            def visit_for_node(node)
                assert_singleton_type(node.range_from.accept(self), Types::PrimitiveType::Int)
                assert_singleton_type(node.range_to.accept(self), Types::PrimitiveType::Int)
                
                changed = symbol_table.ensure_variable_declared(node.iterator_identifier, type: Types::UnionType.create_int)
                
                super(node)
                
                # TODO: Should return range

                node.get_type.expand_return_type(Types::UnionType.create_int)
            end
            
            def visit_break_node(node)
                Types::UnionType.create_void
            end
            
            def visit_if_node(node)
                assert_singleton_type(node.condition.accept(self), Types::PrimitiveType::Bool)
                
                type = Types::UnionType.new
                type.expand(node.true_body_stmts.accept(self))       # Begin always has type of last stmt

                if node.false_body_stmts == nil
                    type.expand(Types::UnionType.create_void)
                else
                    type.expand(node.false_body_stmts.accept(self))
                end

                node.get_type.expand_return_type(type)
            end
            
            def visit_begin_node(node)
                node.body_stmts[0...-1].each do |stmt|
                    stmt.accept(self)
                end
                
                # TODO: need to handle empty BeginNode?
                type = node.body_stmts.last.accept(self)
                node.get_type.expand_return_type(type)
            end
            
            def visit_return_node(node)
                type = node.value.accept(self)
                symbol_table.expand_return_type(type)
                node.get_type.expand_return_type(type)
            end


            ArithOperators = [:+, :-, :*, :/, :%]
            CompareOperators = [:<, :<=, :>, :>=]
            EqualityOperators = [:==, :!=]
            LogicOperators = [:&, :'&&', :|, :'||', :^]
            PrimitiveOperators = ArithOperators + CompareOperators + EqualityOperators + LogicOperators
                
            def visit_send_node(node)
                # TODO: handle self sends
                receiver_type = nil

                if node.receiver == nil
                    receiver_type = Types::UnionType.create_int
                else
                    receiver_type = node.receiver.accept(self)
                end
                type = Types::UnionType.new
                
                if PrimitiveOperators.include?(node.selector)
                    if node.arguments.size != 1
                        raise "Expected 1 argument for binary selector (#{node.arguments.size} given)"
                    end
                    
                    # Process every combination of recv_type x operand_type
                    operand_type = node.arguments.first.accept(self)
                    for recv_type in receiver_type.types
                        for op_type in operand_type.types
                            type.expand(primitive_operator_type(node.selector, recv_type, op_type))
                        end
                    end
                else
                    for recv_type in receiver_type.types
                        if recv_type.to_ruby_type.singleton_methods.include?(("_ikra_t_" + node.selector.to_s).to_sym)
                            # TODO: pass arguments
                            type.expand(recv_type.to_ruby_type.send(("_ikra_t_" + node.selector.to_s).to_sym, receiver_type))
                        else
                            node.arguments.each do |arg|
                                arg.accept(self)
                            end

                            type.expand(visit_method_call(node))
                        end
                    end
                end
                
                node.get_type.expand_return_type(type)
            end

            def primitive_operator_type(selector, receiver_type, operand_type)
                # receiver_type and operand_type are singleton types, return value is union type

                arg_types = [receiver_type, operand_type]
                
                if ArithOperators.include?(selector)
                    type_mapping = {[Types::PrimitiveType::Int, Types::PrimitiveType::Int] => Types::UnionType.create_int,
                        [Types::PrimitiveType::Int, Types::PrimitiveType::Float] => Types::UnionType.create_float,
                        [Types::PrimitiveType::Float, Types::PrimitiveType::Float] => Types::UnionType.create_float}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif CompareOperators.include?(selector)
                    type_mapping = {[Types::PrimitiveType::Int, Types::PrimitiveType::Int] => Types::UnionType.create_bool,
                        [Types::PrimitiveType::Int, Types::PrimitiveType::Float] => Types::UnionType.create_bool,
                        [Types::PrimitiveType::Float, Types::PrimitiveType::Float] => Types::UnionType.create_bool}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif EqualityOperators.include?(selector)
                    type_mapping = {[Types::PrimitiveType::Bool, Types::PrimitiveType::Bool] => Types::UnionType.create_bool,
                        [Types::PrimitiveType::Int, Types::PrimitiveType::Int] => Types::UnionType.create_bool,
                        [Types::PrimitiveType::Int, Types::PrimitiveType::Float] => Types::UnionType.create_bool,
                        [Types::PrimitiveType::Float, Types::PrimitiveType::Float] => Types::UnionType.create_bool}
                        
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    elsif not arg_types.include?(Types::PrimitiveType::Void) and receiver.type.is_primitive? and operand.type.is_primitive?
                        # TODO: this should also return a translation result: selector == :== ? "false" : "true"
                        return Types::UnionType.create_bool
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif LogicOperators.include?(selector)
                    # TODO: need proper implementation
                    int_float = [Types::PrimitiveType::Int, Types::PrimitiveType::Float].to_set
                    if selector == :'&&'
                        if (int_float + arg_types).size == 2
                            # Both are int/float
                            # TODO: this should return the operand
                            return Types::UnionType.new(operand_type)
                        elsif operand_type == PrimitiveType::Bool and receiver_type == Types::PrimitiveType::Bool
                            return Types::UnionType.create_bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :'||'
                        if (int_float + arg_types).size == 2
                            # Both are int/float
                            # TODO: this should return the receiver
                            return Types::UnionType.new(receiver_type)
                        elsif operand_type == Types::PrimitiveType::Bool and receiver_type == Types::PrimitiveType::Bool
                            return Types::UnionType.create_bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :& or selector == :| or selector == :^
                        type_mapping = {[Types::PrimitiveType::Bool, Types::PrimitiveType::Bool] => Types::UnionType.create_bool,
                            [Types::PrimitiveType::Int, Types::PrimitiveType::Int] => Types::UnionType.create_int}
                            
                        if type_mapping.has_key?(arg_types)
                            return type_mapping[arg_types]
                        else
                            raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                        end
                    end
                end
            end
        end
    end
end
