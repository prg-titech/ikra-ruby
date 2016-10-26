require_relative "../ast/nodes.rb"

# Rule: every statement ends with newline
# TODO: Add proper exceptions (CompilationError)

module Ikra
    module AST
        class TreeNode
            @@next_temp_identifier_id = 0

            def translate_statement
                translate_expression + ";\n"
            end
            
            def translate_expression
                return_value = expression_return_value
                inner_stmts = nil

                if return_value != nil
                    # Node specifies a return value
                    inner_stmts = translate_statement + "return " + return_value + ";\n"
                else
                    inner_stmts = translate_statement
                end

                statements_as_expression(inner_stmts)
            end

            protected
            
            def expression_return_value
                "NULL"
            end

            def statements_as_expression(str)
                "[&]#{wrap_in_c_block(str, omit_newl: true)}()"
            end
            
            def indent_block(str)
                str.split("\n").map do |line| "    " + line end.join("\n")
            end

            def wrap_in_c_block(str, omit_newl: false)
                result = "{\n" + indent_block(str) + "\n}"

                if omit_newl
                    return result
                else
                    return result + "\n"
                end
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
        
        class BlockDefNode
            def translate_block
                body.translate_statement
            end
        end

        class MethDefNode
            def translate_method
                # TODO: merge with BlockTranslator
                
                method_params = (["environment_t * #{Translator::Constants::ENV_IDENTIFIER}", "#{parent.get_type.to_c_type} #{Constants::SELF_IDENTIFIER}"] + parameters_names_and_types.map do |name, type|
                    "#{name} #{type.singleton_type.to_c_type}"
                end).join(", ")

                # TODO: load environment variables

                # Declare local variables
                local_variables_def = ""
                local_variables_names_and_types.each do |name, type|
                    local_variables_def += "#{types.to_c_type} #{name};\n"
                end

                signature = "__device__ #{get_type.singleton_type.to_c_type} #{parent.get_type.mangled_method_name(name)}(#{method_params})"
                signature + "\n" + Translator.wrap_in_c_block(local_variables_def + body.translate_statement)
            end
        end

        class BehaviorNode
            def translate_statement
                raise "Methods/blocks cannot be translated as a statement"
            end

            def translate_expresion
                raise "Methods/blocks cannot be translated as an expression"
            end
        end

        class ConstNode
            def translate_expression
                raise "Not implemented"
            end
        end

        class RootNode
            def translate_statement
                single_child.translate_statement
            end
        end

        class LVarReadNode
            def translate_expression
                mangled_identifier.to_s
            end
        end
        
        class LVarWriteNode
            def translate_expression
                "#{mangled_identifier.to_s} = #{value.translate_expression}"
            end
        end
        
        class IVarReadNode
            def translate_expression
                array_identifier = enclosing_class.ruby_class.to_ikra_type.inst_var_array_name(identifier)
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
        end
        
        class WhileNode
            def translate_statement
                "while (#{condition.translate_expression})\n#{body_stmts.translate_statement}"
            end
        end

        class BreakNode
            def translate_expression
                raise "BreakNode is never an expression"
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
                # Make every branch return
                accept(Translator::LastStatementReturnsVisitor.new)

                # Wrap in StatementExpression
                statements_as_expression(translate_statement)
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

            def translate_expression
                if body_stmts.size == 0
                    raise "Empty BeginNode cannot be an expression"
                elsif body_stmts.size == 1
                    # Preserve brackets
                    "(#{body_stmts.first.translate_expression})"
                else
                    # Wrap in lambda
                    # Do not worry about scope of varibles, they will all be declared at the beginning of the function
                    accept(Translator::LastStatementReturnsVisitor.new)
                    statements_as_expression(translate_statement)
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
                            if receiver.get_type.singleton_type.should_generate_self_arg?
                                self_argument = [receiver.translate_expression]
                            else
                                self_argument = ["NULL"]
                            end

                            args = ([Translator::Constants::ENV_IDENTIFIER] + self_argument) +
                                arguments.map do |arg| arg.translate_expression end
                            args_string = args.join(", ")

                            "#{receiver.get_type.singleton_type.mangled_method_name(selector)}(#{args_string})"
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
                                if type.should_generate_self_arg?
                                    # No need to pass type as subtypes are regarded as entirely new types
                                    self_argument = ["#{receiver_identifier}.object_id"]
                                else
                                    self_argument = ["NULL"]
                                end

                                args = ([Translator::Constants::ENV_IDENTIFIER] + self_argument) +
                                    arguments.map do |arg| arg.translate_expression end
                                args_string = args.join(", ")
                                
                                case_statements.push("case #{type.class_id}: #{result_identifier} = #{type.mangled_method_name(selector)}(#{args_string}); break;")
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