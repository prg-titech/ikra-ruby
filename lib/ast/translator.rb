require_relative "nodes.rb"
require_relative "../translator/translator"

# Rule: every statement ends with newline

module Ikra
    module AST
        class Node
            @@next_temp_identifier_id = 0

            def translate_statement
                translate_expression + ";\n"
            end
            
            protected
            
            def statements_as_expression(str)
                "[&]{ #{str} }()"
            end
            
            def indent_block(str)
                str.split("\n").map do |line| "    " + line end.join("\n")
            end

            def wrap_in_c_block(str)
                "{\n" + indent_block(str) + "\n}\n"
            end

            def temp_identifier_id
                @@next_temp_identifier_id += 1
                @@next_temp_identifier_id
            end

            # Generates code that assigns the value of a node to a newly-defined variable.
            def define_assign_variable(name, node)
                type = node.get_type.to_c_type
                "#{type} #{name} = #{node.translate_expression};"
            end
        end
        
        class MethodOrBlockNode
            def translate_statement
                child.translate_statement
            end

            def translate_expression
                child.translate_expression
            end
        end

        class ConstNode
            def translate_expression
                raise "Not implemented"
            end
        end

        class LVarReadNode
            def translate_expression
                identifier.to_s
            end
        end
        
        class LVarWriteNode
            def translate_expression
                "#{identifier.to_s} = #{value.translate_expression}"
            end
        end
        
        class IVarReadNode
            def translate_expression
                array_identifier = class_owner.to_ikra_type.inst_var_array_name(identifier)
                "#{Translator::Constants::ENV_IDENTIFIER}->#{array_identifier}[#{Constants::SELF_IDENTIFIER}]"
            end
        end

        class IntNode
            def translate_expression
                value.to_s
            end
        end
        
        class FloatNode
            def translate_expression
                value.to_s
            end
        end
        
        class BoolNode
            def translate_expression
                value.to_s
            end
        end
        
        class ForNode
            def translate_statement
                loop_header = "for (#{iterator_identifier.to_s} = #{range_from.translate_expression}; #{iterator_identifier.to_s} <= #{range_to.translate_expression}; #{iterator_identifier.to_s}++)"
                loop_header + "\n" + body_stmts.translate_statement + "#{iterator_identifier.to_s}--;\n"
            end
            
            def translate_expression
                # TODO: return value should be range
                loop_header = "for (#{iterator_identifier.to_s} = #{range_from.translate_expression}; #{iterator_identifier.to_s} <= #{range_to.translate_expression}; #{iterator_identifier.to_s}++)"
                full_loop = loop_header + "\n" + body_stmts.translate_statement + "#{iterator_identifier.to_s}--;\nreturn 0;"
                statements_as_expression(full_loop)
            end
        end
        
        class BreakNode
            def translate_expression
                raise "Not implemented yet"
            end
            
            def translate_statement
                "break;\n"
            end
        end
        
        class IfNode
            def translate_statement
                header = "if (#{condition.translate_expression})\n"
                
                if false_body_stmts == nil
                    header + true_body_stmts.translate_statement
                else
                    header + true_body_stmts.translate_statement + "else\n" + false_body_stmts.translate_statement
                end
            end
            
            def translate_expression
                header = "if (#{condition.translate_expression})\n"
                true_body_translated = nil
                false_body_translated = nil
                
                if true_body_stmts.is_begin_node?
                    true_body_translated = true_body_stmts.translate_statement_last_returns
                else
                    true_body_translated = "return " + true_body_stmts.translate_expression + ";\n"
                end
                
                if false_body_stmts != nil
                    # Can be begin node or anything else
                    if false_body_stmts.is_begin_node?
                        false_body_translated = false_body_stmts.translate_statement_last_returns
                    else
                        false_body_translated = "return " + false_body_stmts.translate_expression + ";\n"
                    end
                end
                
                if false_body_translated == nil
                    statements_as_expression(header + true_body_translated)
                else
                    statements_as_expression(header + true_body_translated + "else\n" + false_body_translated)
                end
            end
        end
        
        class BeginNode
            def translate_statement
                if body_stmts.size == 0
                    return ""
                end
                
                body_translated = body_stmts.map do |stmt|
                    stmt.translate_statement
                end.join("")
                
                wrap_in_c_block(body_translated)
            end
            
            def translate_statement_last_returns
                if body_stmts.size == 0
                    raise "Cannot return empty BeginNode"
                end
                
                body_translated = BeginNode.new(body_stmts[0...-1]).translate_statement
                body_translated + "return #{body_stmts.last.translate_expression};\n"
                wrap_in_c_block(body_translated)
            end
            
            def translate_expression
                if body_stmts.size == 0
                    raise "Empty BeginNode cannot be an expression"
                elsif body_stmts.size == 1
                    # Preserve brackets
                    "(#{body_stmts.first.translate_expression})"
                else
                    # Wrap in lambda
                    # Do not worry about scope of varibles, they will all be declared at the beginning of the function
                    statements_as_expression(translate_statement_last_returns)
                end
            end
        end
        
        class SendNode
            BinarySelectors = [:+, :-, :*, :/, :%, :<, :<=, :>, :>=, :==, :!=, :&, :'&&', :|, :'||', :^]
            
            def translate_expression
                if BinarySelectors.include?(selector)
                    if arguments.size != 1
                        raise "Expected 1 argument for binary selector (#{arguments.size} given)"
                    end
                    
                    "(#{receiver.translate_expression} #{selector.to_s} #{arguments.first.translate_expression})"
                else
                    if receiver.get_type.is_singleton? and
                            receiver.get_type.singleton_type.to_ruby_type.singleton_methods.include?(("_ikra_c_" + selector.to_s).to_sym)
                        # TODO: support multiple types for receiver
                        receiver.get_type.singleton_type.to_ruby_type.send(("_ikra_c_" + selector.to_s).to_sym, receiver.translate_expression)
                    else
                        # TODO: generate argument code only once

                        if receiver.get_type.is_singleton?
                            self_argument = []
                            if receiver.get_type.singleton_type.should_generate_type?
                                self_argument = [receiver]
                            end

                            args = ([Translator::Constants::ENV_IDENTIFIER] + (self_argument + arguments).map do |arg|
                                arg.translate_expression
                            end).join(", ")
                            
                            "#{receiver.get_type.singleton_type.mangled_method_name(selector)}(#{args})"
                        else
                            # Polymorphic case
                            # TODO: This is not an expression anymore!
                            poly_id = temp_identifier_id
                            receiver_identifier = "_polytemp_recv_#{poly_id}"
                            result_identifier = "_polytemp_result_#{poly_id}"
                            header = "#{define_assign_variable(receiver_identifier, receiver)}\n#{get_type.to_c_type} #{result_identifier};\nswitch (#{receiver_identifier}.class_id)\n"
                            case_statements = []

                            receiver.get_type.types.each do |type|
                                self_argument = []
                                if type.should_generate_type?
                                    # No need to pass type as subtypes are regarded as entirely new types
                                    self_argument = ["#{receiver_identifier}.object_id"]
                                end

                                args = ([Translator::Constants::ENV_IDENTIFIER] + self_argument + arguments.map do |arg|
                                    arg.translate_expression
                                end).join(", ")
                                
                                case_statements.push("case #{type.class_id}: #{result_identifier} = #{type.mangled_method_name(selector)}(#{args}); break;")
                            end

                            # TODO: compound statements only work with the GNU C++ compiler
                            "(" + wrap_in_c_block(header + wrap_in_c_block(case_statements.join("\n")) + result_identifier + ";")[0..-2] + ")"
                        end
                    end
                end
            end
        end

        class ReturnNode
            def translate_expression
                raise "ReturnNode is never an expression"
            end

            def translate_statement
                "return #{value.translate_expression};\n"
            end
        end
    end
end