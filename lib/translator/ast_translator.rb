require_relative "../ast/nodes.rb"
require_relative "../ast/visitor.rb"
require_relative "../ruby_core/ruby_integration"

# Rule: every statement ends with newline
# TODO: Add proper exceptions (CompilationError)

module Ikra
    module Translator
        class ASTTranslator < AST::Visitor
            class ExpressionTranslator
                attr_reader :translator

                def initialize(translator)
                    @translator = translator
                end

                def expression_translator
                    return self
                end

                def statement_translator
                    return translator.statement_translator
                end

                def method_missing(symbol, *args)
                    if symbol.to_s.start_with?("visit_")
                        if statement_translator.respond_to?(symbol)
                            return statements_as_expression(
                                statement_translator.send(symbol, *args) + "return NULL;")
                        else
                            super
                        end
                    else
                        return translator.send(symbol, *args)
                    end
                end

                def visit_behavior_node(node)
                    raise "Methods/blocks cannot be translated as an expression"
                end

                def visit_source_code_expr_node(node)
                    return node.code
                end

                def visit_const_node(node)
                    raise NotImplementedError.new
                end

                def visit_lvar_read_node(node)
                    return node.mangled_identifier.to_s
                end

                def visit_lvar_write_node(node)
                    if node.value.get_type.is_singleton? and !node.symbol_table[node.identifier].is_singleton?
                        # The assigned value is singleton, but the variable is not
                        singleton_assignment = wrap_in_union_type(
                            node.value.accept(expression_translator), 
                            node.value.get_type.singleton_type)
                        return Translator.read_file(file_name: "ast/assignment.cpp",
                            replacements: { 
                                "source" => singleton_assignment,
                                "target" => node.mangled_identifier.to_s})
                    else
                        return Translator.read_file(file_name: "ast/assignment.cpp",
                            replacements: { 
                                "source" => node.value.accept(expression_translator),
                                "target" => node.mangled_identifier.to_s})
                    end
                end

                def visit_ivar_read_node(node)
                    array_identifier = node.enclosing_class.ruby_class.to_ikra_type.inst_var_array_name(identifier)
                    return "#{Constants::ENV_IDENTIFIER}->#{array_identifier}[#{Constants::SELF_IDENTIFIER}]"
                end

                def visit_int_node(node)
                    return node.value.to_s
                end

                def visit_nil_node(node)
                    return "0"
                end

                def visit_float_node(node)
                    return node.value.to_s
                end

                def visit_bool_node(node)
                    return node.value.to_s
                end

                def visit_if_node(node)
                    # Make every branch return
                    node.accept(LastStatementReturnsVisitor.new)

                    # Wrap in StatementExpression
                    return statements_as_expression(node.accept(statement_translator))
                end

                def visit_ternary_node(node)
                    return "((#{node.condition.accept(expression_translator)}) ? (#{node.true_val.accept(expression_translator)}) : (#{node.false_val.accept(expression_translator)}))"
                end


                def visit_begin_node(node)
                    if node.body_stmts.size == 0
                        raise "Empty BeginNode cannot be an expression"
                    elsif node.body_stmts.size == 1
                        # Preserve brackets
                        return "(#{node.body_stmts.first.accept(self)})"
                    else
                        # Wrap in lambda
                        # Do not worry about scope of varibles, they will all be declared at the
                        # beginning of the function
                        node.accept(LastStatementReturnsVisitor.new)
                        return statements_as_expression(node.accept(statement_translator))
                    end
                end

                # Builds a synthetic [AST::SourceCodeExprNode] with a type and a translation.
                def build_synthetic_code_node(code, type)
                    node = AST::SourceCodeExprNode.new(code: code)
                    node.get_type.expand_return_type(type.to_union_type)
                    return node
                end

                def visit_send_node(node)
                    if node.receiver.get_type.is_singleton?
                        return generate_send_for_singleton(
                            node, 
                            node.receiver,
                            node.get_type)
                    else
                        # Polymorphic case
                        # TODO: This is not an expression anymore!
                        poly_id = temp_identifier_id
                        receiver_identifier = "_polytemp_recv_#{poly_id}"
                        result_identifier = "_polytemp_result_#{poly_id}"
                        header = "#{define_assign_variable(receiver_identifier, node.receiver)}\n#{node.get_type.to_c_type} #{result_identifier};\nswitch (#{receiver_identifier}.class_id)\n"
                        case_statements = []

                        for type in node.receiver.get_type
                            if type == Types::PrimitiveType::Int
                                self_node = build_synthetic_code_node(
                                    "#{receiver_identifier}.value.int_", type)
                            elsif type == Types::PrimitiveType::Float
                                self_node = build_synthetic_code_node(
                                    "#{receiver_identifier}.value.float_", type)
                            elsif type == Types::PrimitiveType::Bool
                                self_node = build_synthetic_code_node(
                                    "#{receiver_identifier}.value.bool_", type)
                            elsif type == Types::PrimitiveType::Nil
                                self_node = build_synthetic_code_node(
                                    "#{receiver_identifier}.value.int_", type)
                            else
                                self_node = build_synthetic_code_node(
                                    "#{receiver_identifier}.value.object_id", type)
                            end

                            singleton_return_type = node.return_type_by_recv_type[type]
                            singleton_invocation = generate_send_for_singleton(
                                node, 
                                self_node,
                                singleton_return_type)

                            if singleton_return_type.is_singleton? and !node.get_type.is_singleton?
                                # The return value of this particular invocation (singleton type 
                                # recv) is singleton, but in general this send can return many 
                                # types
                                singleton_invocation = wrap_in_union_type(
                                    singleton_invocation, 
                                    singleton_return_type.singleton_type)
                            end

                            case_statements.push("case #{type.class_id}: #{result_identifier} = #{singleton_invocation}; break;")
                        end

                        # TODO: compound statements only work with the GNU C++ compiler
                        return "(" + wrap_in_c_block(
                            header + 
                            wrap_in_c_block(case_statements.join("\n")) + 
                                result_identifier + ";")[0..-2] + ")"
                    end
                end

                def generate_send_for_singleton(node, singleton_recv, return_type)
                    recv_type = singleton_recv.get_type.singleton_type

                    if RubyIntegration.has_implementation?(recv_type, node.selector)
                        return RubyIntegration.get_implementation(
                            singleton_recv,
                            node.selector, 
                            node.arguments, 
                            translator,
                            return_type)
                    elsif recv_type.is_a?(Types::StructType)
                        first_arg = node.arguments.first

                        if first_arg.is_a?(AST::IntLiteralNode)
                            # Reading the struct at a constant position
                            return recv_type.generate_read(
                                singleton_recv.accept(self), 
                                node.selector, 
                                first_arg.accept(self))
                        else
                            # Reading the struct at a non-constant position
                            id = temp_identifier_id
                            name = "_temp_var_#{id}"
                            first_arg_eval = first_arg.accept(self)

                            # Store index in local variable, then generate non-constant access
                            # TODO: Statement expression is potentially inefficient
                            return "({ int #{name} = #{first_arg_eval};\n" +
                                recv_type.generate_non_constant_read(
                                    singleton_recv.accept(self),
                                    node.selector,
                                    first_arg_eval) + "; })"
                        end
                    else
                        args = [Constants::ENV_IDENTIFIER]

                        if recv_type.should_generate_self_arg?
                            args.push(singleton_recv.accept(self))
                        else
                            args.push("NULL")
                        end

                        args.push(*(node.arguments.map do |arg| arg.accept(self) end))
                        args_string = args.join(", ")

                        return "#{node.receiver.get_type.singleton_type.mangled_method_name(node.selector)}(#{args_string})"
                    end
                end

                def visit_return_node(node)
                    raise "ReturnNode is never an expression"
                end
            end

            class StatementTranslator
                attr_reader :translator

                def initialize(translator)
                    @translator = translator
                end

                def expression_translator
                    return translator.expression_translator
                end

                def statement_translator
                    return self
                end

                def method_missing(symbol, *args)
                    if symbol.to_s.start_with?("visit_")
                        if expression_translator.respond_to?(symbol)
                            return expression_translator.send(symbol, *args) + ";\n"
                        else
                            super
                        end
                    else
                        return translator.send(symbol, *args)
                    end
                end

                def visit_behavior_node(node)
                    raise "Methods/blocks cannot be translated as a statement"
                end

                def visit_root_node(node)
                    return node.single_child.accept(self)
                end

                def visit_for_node(node)
                    loop_header = "for (#{node.iterator_identifier.to_s} = #{node.range_from.accept(expression_translator)}; #{node.iterator_identifier.to_s} <= #{node.range_to.accept(expression_translator)}; #{node.iterator_identifier.to_s}++)"

                    return loop_header + 
                        "\n" + 
                        node.body_stmts.accept(self) + 
                        "#{node.iterator_identifier.to_s}--;\n"
                end

                def visit_while_node(node)
                    return "while (#{node.condition.accept(expression_translator)})\n#{node.body_stmts.accept(self)}"
                end

                def visit_while_post_node(node)
                    return "do #{node.body_stmts.accept(self)}while (#{node.condition.accept(expression_translator)});\n"
                end

                def visit_until_node(node)
                    return "while (#{node.condition.accept(expression_translator)})\n#{node.body_stmts.accept(self)}"
                end

                def visit_until_post_node(node)
                    return "do #{node.body_stmts.accept(self)}while (#{node.condition.accept(expression_translator)});\n"
                end

                def visit_break_node(node)
                    return "break;\n"
                end

                def visit_if_node(node)
                    header = "if (#{node.condition.accept(expression_translator)})\n"

                    if node.false_body_stmts == nil
                        return header + node.true_body_stmts.accept(self)
                    else
                        return header + node.true_body_stmts.accept(self) + "else\n" + node.false_body_stmts.accept(self)
                    end
                end

                def visit_begin_node(node)
                    if node.body_stmts.size == 0
                        return wrap_in_c_block("")
                    end
                    
                    body_translated = node.body_stmts.map do |stmt|
                        stmt.accept(self)
                    end.join("")

                    return wrap_in_c_block(body_translated)
                end

                def visit_return_node(node)
                    return "return #{node.value.accept(expression_translator)};\n"
                end
            end

            attr_reader :expression_translator

            attr_reader :statement_translator

            def initialize
                @expression_translator = ExpressionTranslator.new(self)
                @statement_translator = StatementTranslator.new(self)
            end

            def statements_as_expression(str)
                return "[&]#{wrap_in_c_block(str, omit_newl: true)}()"
            end
            
            def indent_block(str)
                return str.split("\n").map do |line| "    " + line end.join("\n")
            end

            def wrap_in_c_block(str, omit_newl: false)
                result = "{\n" + indent_block(str) + "\n}"

                if omit_newl
                    return result
                else
                    return result + "\n"
                end
            end

            @@next_temp_identifier_id = 0
            def temp_identifier_id
                @@next_temp_identifier_id += 1
                @@next_temp_identifier_id
            end

            # Generates code that assigns the value of a node to a newly-defined variable.
            def define_assign_variable(name, node)
                type = node.get_type.to_c_type
                return "#{type} #{name} = #{node.accept(expression_translator)};"
            end

            def wrap_in_union_type(str, type)
                if type == Types::PrimitiveType::Int
                    return "((union_t) {#{type.class_id}, {.int_ = #{str}}})"
                elsif type == Types::PrimitiveType::Float
                    return "((union_t) {#{type.class_id}, {.float_ = #{str}}})"
                elsif type == Types::PrimitiveType::Bool
                    return "((union_t) {#{type.class_id}, {.bool_ = #{str}}})"
                elsif type == Types::PrimitiveType::Nil
                    return "((union_t) {#{type.class_id}, {.int_ = #{str}}})"
                elsif !type.is_a?(Types::UnionType)
                    return "((union_t) {#{type.class_id}, {.object_id = #{str}}})"
                else
                    raise "UnionType found but singleton type expected"
                end
            end

            def self.translate_block(block_def_node)
                return self.new.translate_block(block_def_node)
            end

            def translate_block(block_def_node)
                return block_def_node.body.accept(statement_translator)
            end

            def self.translate_method(method_def_node)
                return self.new.translate_method(method_def_node)
            end

            def translate_method(meth_def_node)
                # TODO: merge with BlockTranslator
                
                method_params = ([
                    "environment_t * #{Constants::ENV_IDENTIFIER}", 
                    "#{meth_def_node.parent.get_type.to_c_type} #{Constants::SELF_IDENTIFIER}"] + 
                        meth_def_node.parameters_names_and_types.map do |name, type|
                            "#{type.singleton_type.to_c_type} #{name}"
                        end).join(", ")

                # TODO: load environment variables

                # Declare local variables
                local_variables_def = ""
                meth_def_node.local_variables_names_and_types.each do |name, type|
                    local_variables_def += "#{type.to_c_type} #{name};\n"
                end

                signature = "__device__ #{meth_def_node.get_type.singleton_type.to_c_type} #{meth_def_node.parent.get_type.mangled_method_name(meth_def_node.name)}(#{method_params})"
                return signature + 
                    "\n" + 
                    wrap_in_c_block(local_variables_def + meth_def_node.body.accept(statement_translator))
            end
        end
    end
end