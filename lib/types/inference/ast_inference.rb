# No explicit `require`s. This file should be includes via types.rb

require "set"

require_relative "../../ast/nodes.rb"
require_relative "../../ast/visitor.rb"
require_relative "clear_types_visitor.rb"

module Ikra
    module AST
        class TreeNode
            def get_type
                @type ||= Types::UnionType.new
                return @type.dup
            end

            def merge_union_type(union_type)
                @type ||= Types::UnionType.new

                if not @type.include_all?(union_type)
                    register_type_change
                end

                return @type.expand_return_type(union_type).dup
            end

            def symbol_table
                return parent.symbol_table
            end

            def register_type_change
                if parent != nil
                    parent.register_type_change
                else
                    # This node is not part of a full AST, i.e., it does not have a [BehaviorNode]
                    # as a parent. Do nothing.
                end
            end
        end

        class ClassDefNode
            def get_type
                return ruby_class.to_ikra_type
            end
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
                    return_type: Types::UnionType.new,
                    method_binding: nil)

                instance = new(name: name, body: body, ruby_method: ruby_method, method_binding: method_binding)
                instance.initialize_types(receiver_type: receiver_type, return_type: return_type)
                instance.parameters_names_and_types = parameters_names_and_types
                return instance
            end

            def callers
                @callers ||= []
            end
        end

        class BehaviorNode
            def get_type
                @type ||= Types::UnionType.new
            end

            def merge_union_type(union_type)
                type = @type ||= Types::UnionType.new

                if not @type.include_all?(union_type)
                    register_type_change
                end
                
                return type.expand_return_type(union_type).dup
            end

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

            def types_changed?
                return @types_changed ||= false
            end

            def reset_types_changed
                @types_changed = false
            end

            def register_type_change
                @types_changed = true
            end
        end

        class SendNode
            attr_writer :return_type_by_recv_type

            # Mapping: Receiver type --> Return value of send
            def return_type_by_recv_type
                @return_type_by_recv_type ||= {}
            end
        end
    end
    
    module TypeInference
        class Visitor < AST::Visitor

            # If this error is thrown, type inference should start from the beginning (with
            # an empty symbol table).
            class RestartTypeInferenceError < RuntimeError

            end

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

                begin
                    # Infer types
                    body_ast.accept(self)
                rescue RestartTypeInferenceError
                    # Reset all type information
                    symbol_table.clear!
                    block_def_node.accept(ClearTypesVisitor.new)

                    # Remove block from stack
                    @work_stack.pop

                    Log.info("Changed AST during type inference. Restarting type inference.")

                    # Restart inference
                    return process_block(block_def_node)
                end

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

                return_type = block_def_node.merge_union_type(return_value_type)

                if block_def_node.types_changed?
                    # Types changed, do another pass. This is not efficient and there are better
                    # ways to do type inference (e.g., constraint solving), but it works for now.
                    block_def_node.reset_types_changed
                    return process_block(block_def_node)
                else
                    return return_type
                end
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

                return_type = method_def_node.merge_union_type(return_value_type)

                if method_def_node.types_changed?
                    # Types changed, do another pass. This is not efficient and there are better
                    # ways to do type inference (e.g., constraint solving), but it works for now.
                    method_def_node.reset_types_changed
                    return process_method(method_def_node)
                else
                    return return_type
                end
            end

            # This is not an actual Visitor method. It is called from visit_send_node.
            def visit_method_call(send_node, recv_singleton_type)
                selector = send_node.selector

                if recv_singleton_type.is_primitive?
                    raise NotImplementedError.new("#{recv_singleton_type}.#{selector} not implemented (#{send_node.to_s})")
                end

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
                            Array.new(arg_types.size) do |arg_index|
                                Types::UnionType.new
                            end).flatten],
                        ruby_method: nil,
                        receiver_type: recv_singleton_type,
                        method_binding: recv_singleton_type.method_binding(selector))
                    @classes[recv_singleton_type].add_instance_method(method_def_node)
                    method_visited_before = false
                else
                    method_visited_before = true
                end

                method_def_node = @classes[recv_singleton_type].instance_method(selector)

                parameter_types_expanded = parameter_names.map.with_index do |name, index|
                    # returns true if expanded 
                    method_def_node.parameters_names_and_types[name].expand(arg_types[index])
                end.reduce(:|)

                # Method needs processing if any parameter is expanded (or method was never visited before)
                needs_processing = !method_visited_before or parameter_types_expanded

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

                return method_def_node.get_type
            end

            def assert_singleton_type(union_type, expected_type)
                if union_type.singleton_type != expected_type
                    raise AssertionError.new(
                        "Expected type #{expected_type} but found #{union_type.singleton_type}")
                end
            end

            def visit_const_node(node)
                if not binding
                    raise AssertionError.new("Unable to resolve constants without Binding")
                end

                constant = binding.eval(node.identifier.to_s)
                node.merge_union_type(constant.ikra_type.to_union_type)
            end

            def visit_root_node(node)
                node.merge_union_type(node.single_child.accept(self))
            end

            def visit_lvar_read_node(node)
                symbol_table.read!(node.identifier)

                # Extend type of variable
                return node.merge_union_type(symbol_table[node.identifier])
            end

            def visit_lvar_write_node(node)
                type = node.value.accept(self)

                # Declare/extend type in symbol table
                symbol_table.ensure_variable_declared(node.identifier, type: type)
                symbol_table.written!(node.identifier)

                node.variable_type = symbol_table[node.identifier]

                # Extend type of variable
                # Note: Return value of this expression != type of the variable
                node.merge_union_type(type)
            end
            
            def visit_ivar_read_node(node)
                cls_type = node.enclosing_class.ruby_class.to_ikra_type
                cls_type.inst_var_read!(node.identifier)
                cls_type.inst_vars_types[node.identifier]
            end

            def visit_int_node(node)
                node.merge_union_type(Types::UnionType.create_int)
            end

            def visit_nil_node(node)
                node.merge_union_type(Types::UnionType.create_nil)
            end
            
            def visit_float_node(node)
                node.merge_union_type(Types::UnionType.create_float)
            end
            
            def visit_bool_node(node)
                node.merge_union_type(Types::UnionType.create_bool)
            end
            
            def visit_string_node(node)
                # Use [Types::ClassType] for the moment
                return node.value.ikra_type.to_union_type
            end

            def visit_symbol_node(node)
                # Use [Types::ClassType] for the moment
                return node.value.ikra_type.to_union_type
            end

            def visit_hash_node(node)
                # Use [Types::ClassType] for the moment
                return Hash.to_ikra_type.to_union_type
            end

            def visit_for_node(node)
                assert_singleton_type(node.range_from.accept(self), Types::PrimitiveType::Int)
                assert_singleton_type(node.range_to.accept(self), Types::PrimitiveType::Int)

                changed = symbol_table.ensure_variable_declared(node.iterator_identifier, type: Types::UnionType.create_int)
                symbol_table.written!(node.iterator_identifier)
                
                super(node)
                
                # TODO: Should return range

                node.merge_union_type(Types::UnionType.create_int)
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

                node.merge_union_type(type)
            end
            
            def visit_ternary_node(node)
                assert_singleton_type(node.condition.accept(self), Types::PrimitiveType::Bool)
                
                type = Types::UnionType.new
                type.expand(node.true_val.accept(self))
                type.expand(node.false_val.accept(self))

                node.merge_union_type(type)
            end
            
            def visit_begin_node(node)
                node.body_stmts[0...-1].each do |stmt|
                    stmt.accept(self)
                end

                if node.body_stmts.empty?
                    type = Types::UnionType.new
                else
                    type = node.body_stmts.last.accept(self)
                end

                node.merge_union_type(type)
            end
            
            def visit_return_node(node)
                type = node.value.accept(self)
                symbol_table.expand_return_type(type)
                node.merge_union_type(type)
            end
                
            def visit_source_code_expr_node(node)
                # This is a synthetic node. No type inference. Return the type that was set
                # manually before (if any).
                return node.get_type
            end

            def visit_send_node_singleton_receiver(sing_type, node)
                if RubyIntegration.is_interpreter_only?(sing_type)
                    return Types::InterpreterOnlyType.new.to_union_type
                elsif RubyIntegration.has_implementation?(sing_type, node.selector)
                    arg_types = node.arguments.map do |arg| arg.get_type end

                    begin
                        return_type = RubyIntegration.get_return_type(
                            sing_type, node.selector, *arg_types, send_node: node)
                        return return_type
                    rescue RubyIntegration::CycleDetectedError => cycle_error
                        # Cannot do further symbolic execution, i.e., kernel fusion here,
                        # because we are in a loop.

                        # Invoke parallel section: change to `RECV` to `RECV.__call__.to_command`
                        node.replace_child(
                            node.receiver, 
                            AST::SendNode.new(
                                receiver: AST::SendNode.new(
                                    receiver: node.receiver, selector: :__call__),
                                selector: :to_command))

                        # Start fresh
                        raise RestartTypeInferenceError.new
                    end
                elsif sing_type.is_a?(Types::StructType)
                    # This is a struct type, special type inference rules apply
                    return sing_type.get_return_type(node.selector, *node.arguments)
                else
                    Log.info("Translate call to ordinary Ruby method #{sing_type}.#{node.selector}")
                    return visit_method_call(node, sing_type)
                end
            end

            def visit_send_node_union_type(receiver_type, node)
                type = Types::UnionType.new

                for sing_type in receiver_type
                    return_type = visit_send_node_singleton_receiver(sing_type, node)
                    node.return_type_by_recv_type[sing_type] = return_type
                    type.expand(return_type)
                end

                node.merge_union_type(type)

                return type
            end

            def visit_send_node(node)
                # TODO: handle self sends
                receiver_type =  node.receiver.accept(self)

                node.arguments.each do |arg|
                    arg.accept(self)
                end
                
                visit_send_node_union_type(receiver_type, node)

                return node.get_type
            end
        end
    end
end
