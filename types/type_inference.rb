require "set"
require_relative "primitive_type"
require_relative "union_type"
require_relative "ruby_extension"
require_relative "../ast/nodes.rb"
require_relative "../ast/visitor.rb"
require_relative "../ast/method_definition"
require_relative "../scope.rb"

module Ikra
    module AST
        class Node
            def get_type
                @type ||= UnionType.new
            end
        end
    end
    
    module TypeInference
        class Visitor < AST::Visitor
            attr_accessor :aux_methods

            def get_aux_method(type, selector, types)
                aux_methods.detect do |aux|
                    type <= aux.type and aux.selector == selector and UnionType.array_subseteq(types, aux.parameter_types)
                end
            end

            def visit_method_call(send_node)
                # TODO: constant lookup (via binding.receiver.eval)
                recv_type = send_node.receiver.get_type
                selector = send_node.selector
                param_types = send_node.arguments.map do |arg|
                    arg.get_type
                end

                cached_value = get_aux_method(recv_type, selector, param_types)
                if cached_value != nil
                    return cached_value.return_type
                end

                Log.info("Type inference: proceed into method #{recv_type.singleton_type}.#{selector}(#{param_types.to_type_array_string})")
                # TODO: handle multiple types
                ast = recv_type.singleton_type.method_ast(selector)

                # Set up new symbol table (pushing a frame is not sufficient here)
                return_value_type = nil
                old_symbol_table = @symbol_table
                @symbol_table = Scope.new
                @symbol_table.new_frame do
                    @symbol_table.top_frame.function_frame!

                    # Add parameters to symbol table (name -> type)
                    recv_type.singleton_type.method_parameters(selector).zip(param_types).each do |param|
                        @symbol_table.declare_expand_type(param[0], param[1])
                    end

                    # Add return statements
                    ast.accept(Ikra::Translator::LastStatementReturnsVisitor.new)

                    # Infer types
                    ast.accept(self)
                    return_value_type = @symbol_table.top_frame.return_type
                end
                
                Log.info("Type inference: method return type is #{return_value_type.to_s}")

                # Restore old symbol table
                @symbol_table = old_symbol_table

                method_def = AST::MethodDefinition.new(
                    type: recv_type.singleton_type, 
                    selector: selector,
                    parameter_types: param_types,
                    return_type: return_value_type,
                    ast: ast)
                @aux_methods.push(method_def)

                method_def.return_type
            end

            def assert_singleton_type(union_type, expected_type)
                if union_type.singleton_type != expected_type
                    raise "Expected type #{expected_type} but found #{union_type.singleton_type}"
                end
            end
            
            def initialize(symbol_table, binding = nil)
                @symbol_table = symbol_table
                @binding = binding

                @aux_methods = []
            end
            
            def visit_root_node(node)
                node.get_type.expand_return_type(node.child.accept(self))
            end

            def visit_const_node(node)
                if not @binding
                    raise "Unable to resolve constants without Binding"
                end

                node.get_type.expand_return_type(
                    UnionType.new([@binding.eval(node.identifier.to_s).class.to_ikra_type]))
            end

            def visit_lvar_read_node(node)
                @symbol_table.read!(node.identifier)
                node.get_type.expand_return_type(@symbol_table.get_type(node.identifier))
            end

            def visit_lvar_write_node(node)
                type = node.value.accept(self)
                @symbol_table.declare_expand_type(node.identifier, type)
                @symbol_table.written!(node.identifier)
                node.get_type.expand_return_type(type)
            end
            
            def visit_int_node(node)
                node.get_type.expand_return_type(UnionType.create_int)
            end
            
            def visit_float_node(node)
                node.get_type.expand_return_type(UnionType.create_float)
            end
            
            def visit_bool_node(node)
                node.get_type.expand_return_type(UnionType.create_bool)
            end
            
            def visit_for_node(node)
                assert_singleton_type(node.range_from.accept(self), PrimitiveType::Int)
                assert_singleton_type(node.range_to.accept(self), PrimitiveType::Int)
                
                changed = @symbol_table.declare_expand_type(node.iterator_identifier, UnionType.create_int)
                
                super(node)
                
                # TODO: Should return range

                node.get_type.expand_return_type(UnionType.create_int)
            end
            
            def visit_break_node(node)
                UnionType.create_void
            end
            
            def visit_if_node(node)
                assert_singleton_type(node.condition.accept(self), PrimitiveType::Bool)
                
                type = UnionType.new
                type.expand(node.true_body_stmts.accept(self))       # Begin always has type of last stmt

                if node.false_body_stmts == nil
                    type.expand(UnionType.create_void)
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
                @symbol_table.add_return_type(type)
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
                    receiver_type = UnionType.create_int
                else
                    receiver_type = node.receiver.accept(self)
                end
                type = UnionType.new
                
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
                    type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => UnionType.create_int,
                        [PrimitiveType::Int, PrimitiveType::Float] => UnionType.create_float,
                        [PrimitiveType::Float, PrimitiveType::Float] => UnionType.create_float}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif CompareOperators.include?(selector)
                    type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => UnionType.create_bool,
                        [PrimitiveType::Int, PrimitiveType::Float] => UnionType.create_bool,
                        [PrimitiveType::Float, PrimitiveType::Float] => UnionType.create_bool}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif EqualityOperators.include?(selector)
                    type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => UnionType.create_bool,
                        [PrimitiveType::Int, PrimitiveType::Int] => UnionType.create_bool,
                        [PrimitiveType::Int, PrimitiveType::Float] => UnionType.create_bool,
                        [PrimitiveType::Float, PrimitiveType::Float] => UnionType.create_bool}
                        
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    elsif not arg_types.include?(PrimitiveType::Void) and receiver.type.is_primitive? and operand.type.is_primitive?
                        # TODO: this should also return a translation result: selector == :== ? "false" : "true"
                        return UnionType.create_bool
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif LogicOperators.include?(selector)
                    # TODO: need proper implementation
                    int_float = [PrimitiveType::Int, PrimitiveType::Float].to_set
                    if selector == :'&&'
                        if (int_float + arg_types).size == 2
                            # Both are int/float
                            # TODO: this should return the operand
                            return UnionType.new(operand_type)
                        elsif operand_type == PrimitiveType::Bool and receiver_type == PrimitiveType::Bool
                            return UnionType.create_bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :'||'
                        if (int_float + arg_types).size == 2
                            # Both are int/float
                            # TODO: this should return the receiver
                            return UnionType.new(receiver_type)
                        elsif operand_type == PrimitiveType::Bool and receiver_type == PrimitiveType::Bool
                            return UnionType.create_bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :& or selector == :| or selector == :^
                        type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => UnionType.create_bool,
                            [PrimitiveType::Int, PrimitiveType::Int] => UnionType.create_int}
                            
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
