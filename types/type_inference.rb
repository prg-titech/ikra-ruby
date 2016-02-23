require "set"
require_relative "primitive_type"
require_relative "ruby_extension"
require_relative "../ast/nodes.rb"
require_relative "../ast/visitor.rb"

module Ikra
    module AST
        class Node
            def add_type(types)
                # TODO: is it safe to return false when assigning type for the first time?
                @types ||= Set.new
                changed = false
                
                if types.class == Array
                    types.each do |type|
                        if not @types.include?(type)
                            changed = true
                            @types.add(type)
                        end
                    end
                else
                    if not @types.include(types)
                        changed = true
                        @types.add(types)
                    end
                end
                
                changed
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
            
            def initialize
                @symbol_table = Scope.new
            end
            
            def visit_lvar_read_node(node)
                types = @symbol_table.get(node.identifier)
                changed = node.add_type(types)
                types
            end
            
            def visit_lvar_write_node(node)
                types = value.accept(self)
                changed = node.add_type(types) ||= @symbol_table.add_type(node.identifier, types)
                types
            end
            
            def visit_int_node(node)
                node.add_type(PrimitiveType::Int)
                [PrimitiveType::Int].to_set
            end
            
            def visit_float_node(node)
                node.add_type(PrimitiveType::Float)
                [PrimitiveType::Float].to_set
            end
            
            def visit_for_node(node)
                assert_single_type(node.range_from.accept(self), PrimitiveType::Int)
                assert_single_type(node.range_to.accept(self), PrimitiveType::Int)
                
                changed = @symbol_table.add_type(node.iterator_identifier, type)
                
                super(node)
                
                # TODO: Should return range
                node.add_type(PrimitiveType::Int)
                [PrimitiveType::Int].to_set
            end
            
            def visit_if_node(node)
                assert_single_type(node.condition.accept(self), PrimitiveType::Bool)
                
                types = Set.new
                types.add(true_body_stmts.accept(self))     # Begin always has type of last stmt
                types.add(false_body_stmts.accept(self))
                
                changed = node.add_type(types)
                types
            end
            
            def visit_begin_node(node)
                node.body_stmts[0...-1].each do |stmt|
                    stmt.accept(self)
                end
                
                # TODO: need to handle empty BeginNode?
                types = node.body_stmts.last.accept(self)
                changed = node.add_type(types)
                types
            end
            
            def begin_send_node(node)
                
            end
        end
    end
end
