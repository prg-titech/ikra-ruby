require "set"
require_relative "primitive_type"
require_relative "ruby_extension"
require_relative "../ast/nodes.rb"
require_relative "../ast/visitor.rb"
require_relative "../scope.rb"

module Ikra
    module AST
        class Node
            def add_types(types)
                # TODO: is it safe to return false when assigning type for the first time?
                @types ||= Set.new
                changed = false
                
                if types.class == Set or types.class == Array
                    types.each do |type|
                        if not @types.include?(type)
                            changed = true
                            @types.add(type)
                        end
                    end
                else
                    if not @types.include?(types)
                        changed = true
                        @types.add(types)
                    end
                end
                
                changed
            end

            def get_types
                @types || Set.new
            end
        end
    end
    
    module TypeInference
        class Visitor < AST::Visitor
            def assert_single_type(type_set, expected_type)
                if type_set.size != 1 || type_set.first != expected_type
                    raise "Expected type #{expected_type} but found #{type_set.to_a}"
                end
            end
            
            def initialize(symbol_table)
                @symbol_table = symbol_table
            end
            
            def visit_lvar_read_node(node)
                types = @symbol_table.get_types(node.identifier)
                @symbol_table.read!(node.identifier)
                changed = node.add_types(types)
                types
            end
            
            def visit_lvar_write_node(node)
                types = node.value.accept(self)
                changed = node.add_types(types)
                @symbol_table.add_types(node.identifier, types)
                @symbol_table.written!(node.identifier)
                types
            end
            
            def visit_int_node(node)
                node.add_types(PrimitiveType::Int)
                [PrimitiveType::Int].to_set
            end
            
            def visit_float_node(node)
                node.add_types(PrimitiveType::Float)
                [PrimitiveType::Float].to_set
            end
            
            def visit_bool_node(node)
                node.add_types(PrimitiveType::Bool)
                [PrimitiveType::Bool].to_set
            end
            
            def visit_for_node(node)
                assert_single_type(node.range_from.accept(self), PrimitiveType::Int)
                assert_single_type(node.range_to.accept(self), PrimitiveType::Int)
                
                changed = @symbol_table.add_types(node.iterator_identifier, [PrimitiveType::Int].to_set)
                
                super(node)
                
                # TODO: Should return range
                node.add_types(PrimitiveType::Int)
                [PrimitiveType::Int].to_set
            end
            
            def visit_break_node(node)
                [PrimitiveType::Void].to_set
            end
            
            def visit_if_node(node)
                assert_single_type(node.condition.accept(self), PrimitiveType::Bool)
                
                types = Set.new
                types.merge(node.true_body_stmts.accept(self))     # Begin always has type of last stmt

                if node.false_body_stmts == nil
                    types.add(PrimitiveType::Void)
                else
                    types.merge(node.false_body_stmts.accept(self))
                end

                changed = node.add_types(types)
                types
            end
            
            def visit_begin_node(node)
                node.body_stmts[0...-1].each do |stmt|
                    stmt.accept(self)
                end
                
                # TODO: need to handle empty BeginNode?
                types = node.body_stmts.last.accept(self)
                changed = node.add_types(types)
                types
            end
            
            def visit_return_node(node)
                types = node.value.accept(self)
                change = node.add_types(types)
                @symbol_table.add_return_types(types)
                types
            end

            ArithOperators = [:+, :-, :*, :/, :%]
            CompareOperators = [:<, :<=, :>, :>=]
            EqualityOperators = [:==, :!=]
            LogicOperators = [:&, :'&&', :|, :'||', :^]
            PrimitiveOperators = ArithOperators + CompareOperators + EqualityOperators + LogicOperators
                
            def visit_send_node(node)
                # TODO: handle self sends
                receiver_types = nil
                if node.receiver == nil
                    receiver_types = [PrimitiveType::Int].to_set
                else
                    receiver_types = node.receiver.accept(self)
                end
                types = Set.new
                
                if PrimitiveOperators.include?(node.selector)
                    if node.arguments.size != 1
                        raise "Expected 1 argument for binary selector (#{node.arguments.size} given)"
                    end
                    
                    operand_types = node.arguments.first.accept(self)
                    for recv_type in receiver_types
                        for op_type in operand_types
                            types.merge([primitive_operator_type(node.selector, recv_type, op_type)].to_set)
                        end
                    end
                else
                    types = Set.new
                    for recv_type in receiver_types
                        if recv_type.to_ruby_type.singleton_methods.include?(("_ikra_t_" + node.selector.to_s).to_sym)
                            # TODO: pass arguments
                            types.merge(recv_type.to_ruby_type.send(("_ikra_t_" + node.selector.to_s).to_sym, receiver_types))
                        else
                            # TODO: handle return value, pass arguments
                            types.merge([PrimitiveType::Void].to_set)
                        end
                    end
                end
                
                
                changed = node.add_types(types)
                types
            end
            
            def primitive_operator_type(selector, receiver_type, operand_type)
                arg_types = [receiver_type, operand_type]
                
                if ArithOperators.include?(selector)
                    type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Int,
                        [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Float,
                        [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Float}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif CompareOperators.include?(selector)
                    type_mapping = {[PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Bool,
                        [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Bool,
                        [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Bool}
                    
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    else
                        raise "Types #{receiver_type} and #{operand_type} not applicable for primitive operator #{selector.to_s}"
                    end
                elsif EqualityOperators.include?(selector)
                    type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => PrimitiveType::Bool,
                        [PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Bool,
                        [PrimitiveType::Int, PrimitiveType::Float] => PrimitiveType::Bool,
                        [PrimitiveType::Float, PrimitiveType::Float] => PrimitiveType::Bool}
                        
                    if type_mapping.has_key?(arg_types)
                        return type_mapping[arg_types]
                    elsif type_mapping.has_key?(arg_types.reverse)
                        return type_mapping[arg_types.reverse]
                    elsif not arg_types.include?(PrimitiveType::Void) and receiver.type.is_primitive? and operand.type.is_primitive?
                        # TODO: this should also return a translation result: selector == :== ? "false" : "true"
                        return PrimitiveType::Bool
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
                            return operand_type
                        elsif operand_type == PrimitiveType::Bool and receiver_type == PrimitiveType::Bool
                            return PrimitiveType::Bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :'||'
                        if (int_float + arg_types).size == 2
                            # Both are int/float
                            # TODO: this should return the receiver
                            return receiver_type
                        elsif operand_type == PrimitiveType::Bool and receiver_type == PrimitiveType::Bool
                            return PrimitiveType::Bool
                        else
                            raise "Cannot handle types #{receiver_type} and #{operand_type} for primitive operator #{selector.to_s}"
                        end
                    elsif selector == :& or selector == :| or selector == :^
                        type_mapping = {[PrimitiveType::Bool, PrimitiveType::Bool] => PrimitiveType::Bool,
                            [PrimitiveType::Int, PrimitiveType::Int] => PrimitiveType::Int}
                            
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
