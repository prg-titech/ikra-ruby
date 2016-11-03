require_relative "../ast/nodes.rb"
require_relative "../ruby_core/ruby_integration"

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

            def wrap_in_union_type(str, type)
                if type == Types::PrimitiveType::Int
                    return "((union_t) {#{type.class_id}, {.int_ = #{str}}})"
                elsif type == Types::PrimitiveType::Float
                    return "((union_t) {#{type.class_id}, {.float_ = #{str}}})"
                elsif type == Types::PrimitiveType::Bool
                    return "((union_t) {#{type.class_id}, {.bool_ = #{str}}})"
                elsif !type.is_a?(Types::UnionType)
                    return "((union_t) {#{type.class_id}, {.object_id = #{str}}})"
                else
                    raise "UnionType found but singleton type expected"
                end
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
                    "#{type.singleton_type.to_c_type} #{name}"
                end).join(", ")

                # TODO: load environment variables

                # Declare local variables
                local_variables_def = ""
                local_variables_names_and_types.each do |name, type|
                    local_variables_def += "#{type.to_c_type} #{name};\n"
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
                if value.get_type.is_singleton? and !symbol_table[identifier].is_singleton?
                    # The assigned value is singleton, but the variable is not
                    singleton_assignment = wrap_in_union_type(
                        value.translate_expression, value.get_type.singleton_type)
                    return "#{mangled_identifier.to_s} = #{singleton_assignment}"
                else
                    return "#{mangled_identifier.to_s} = #{value.translate_expression}"
                end
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
        
        class WhilePostNode
            def translate_statement
                "do #{body_stmts.translate_statement}while (#{condition.translate_expression});\n"
            end
        end
        
        class UntilNode
            def translate_statement
                "while (#{condition.translate_expression})\n#{body_stmts.translate_statement}"
            end
        end
        
        class UntilPostNode
            def translate_statement
                "do #{body_stmts.translate_statement}while (#{condition.translate_expression});\n"
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
            def translate_expression
                if receiver.get_type.is_singleton?
                    return generate_send_for_singleton(receiver.get_type.singleton_type)
                else
                    # Polymorphic case
                    # TODO: This is not an expression anymore!
                    poly_id = temp_identifier_id
                    receiver_identifier = "_polytemp_recv_#{poly_id}"
                    result_identifier = "_polytemp_result_#{poly_id}"
                    header = "#{define_assign_variable(receiver_identifier, receiver)}\n#{get_type.to_c_type} #{result_identifier};\nswitch (#{receiver_identifier}.class_id)\n"
                    case_statements = []

                    for type in receiver.get_type.types
                        object_id = nil

                        if type == Types::PrimitiveType::Int
                            object_id = "#{receiver_identifier}.value.int_"
                        elsif type == Types::PrimitiveType::Float
                            object_id = "#{receiver_identifier}.value.float_"
                        elsif type == Types::PrimitiveType::Bool
                            object_id = "#{receiver_identifier}.value.bool_"
                        else
                            object_id = "#{receiver_identifier}.value.object_id"
                        end

                        singleton_invocation = generate_send_for_singleton(type, self_argument: object_id)
                        singleton_return_value = return_type_by_recv_type[type]

                        if singleton_return_value.is_singleton? and !get_type.is_singleton?
                            # The return value of this particular invocation (singleton type recv)
                            # is singleton, but in general this send can return many types
                            singleton_invocation = wrap_in_union_type(singleton_invocation, singleton_return_value.singleton_type)
                        end

                        case_statements.push("case #{type.class_id}: #{result_identifier} = #{singleton_invocation}; break;")
                    end

                    # TODO: compound statements only work with the GNU C++ compiler
                    "(" + wrap_in_c_block(header + wrap_in_c_block(case_statements.join("\n")) + result_identifier + ";")[0..-2] + ")"
                end
            end

            def generate_send_for_singleton(recv_type, self_argument: nil)
                ruby_recv_type = recv_type.to_ruby_type

                if RubyIntegration.has_implementation?(ruby_recv_type, selector)
                    args = []

                    if RubyIntegration.should_pass_self?(ruby_recv_type, selector)
                        if self_argument != nil
                            args.push(self_argument)
                        else
                            args.push(receiver.translate_expression)
                        end
                    end

                    # Add regular arguments
                    args.push(*arguments.map do |arg| arg.translate_expression end)

                    return RubyIntegration.get_implementation(ruby_recv_type, selector, *args)
                else
                    args = [Translator::Constants::ENV_IDENTIFIER]

                    if recv_type.should_generate_self_arg?
                        if self_argument != nil
                            args.push(self_argument) 
                        else
                            args.push(receiver.translate_expression)
                        end
                    else
                        args.push("NULL")
                    end

                    args.push(*(arguments.map do |arg| arg.translate_expression end))
                    args_string = args.join(", ")

                    return "#{receiver.get_type.singleton_type.mangled_method_name(selector)}(#{args_string})"
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