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

                def generate_polymorphic_switch(node, &block)
                    poly_id = temp_identifier_id
                    node_identifer = "_polytemp_expr_#{poly_id}"
                    header = "#{define_assign_variable(node_identifer, node)}\nswitch (#{node_identifer}.class_id)\n"
                    case_statements = []

                    for type in node.get_type
                        if type == Types::PrimitiveType::Int
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.int_", type)
                        elsif type == Types::PrimitiveType::Float
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.float_", type)
                        elsif type == Types::PrimitiveType::Bool
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.bool_", type)
                        elsif type == Types::PrimitiveType::Nil
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.int_", type)
                        elsif type.is_a?(Symbolic::ArrayCommand)
                            self_node = build_synthetic_code_node(
                                "(#{type.to_c_type}) #{node_identifer}.value.array_command", type)
                        elsif type.is_a?(Types::LocationAwareFixedSizeArrayType)
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.fixed_size_array", type)
                        else
                            self_node = build_synthetic_code_node(
                                "#{node_identifer}.value.object_id", type)
                        end

                        case_statements.push("case #{type.class_id}: #{yield(self_node)} break;")
                    end

                    return header + wrap_in_c_block(case_statements.join("\n"))
                end

                def visit_send_node(node)
                    if node.receiver.get_type.is_singleton?
                        return_type = node.return_type_by_recv_type[
                            node.receiver.get_type.singleton_type]

                        invocation = generate_send_for_singleton(
                            node, 
                            node.receiver,
                            return_type)

                            if return_type.is_singleton? and
                                !node.get_type.is_singleton?

                                invocation = wrap_in_union_type(
                                    invocation, 
                                    return_type.singleton_type)
                            end

                            return invocation
                    else
                        # Polymorphic case
                        result_identifier = "_polytemp_result_#{temp_identifier_id}"
                        declare_result_var = "#{node.get_type.to_c_type} #{result_identifier};\n"

                        case_statements = generate_polymorphic_switch(node.receiver) do |self_node|
                            # The singleton type in the current case
                            type = self_node.get_type.singleton_type

                            # The return type (result type) in the current case (could be polym.)
                            return_type = node.return_type_by_recv_type[type]

                            # Generate method invocation
                            invocation = generate_send_for_singleton(
                                node, 
                                self_node,
                                return_type)

                            if return_type.is_singleton? and
                                !node.get_type.is_singleton?
                                # The return value of this particular invocation (singleton type 
                                # recv) is singleton, but in general this send can return many 
                                # types
                                invocation = wrap_in_union_type(
                                    invocation, 
                                    return_type.singleton_type)
                            end

                            "#{result_identifier} = #{invocation};"
                        end

                        # TODO: compound statements only work with the GNU C++ compiler
                        return "(" + wrap_in_c_block(
                            declare_result_var + 
                            wrap_in_c_block(case_statements) + 
                                result_identifier + ";")[0..-2] + ")"
                    end
                end

                def build_switch_for_args(nodes, accumulator = [], &block)
                    if nodes.size == 0
                        # This was the last argument, we are done with nesting switch 
                        # stmts. The accumulator contains all singleton-typed self_nodes.
                        return yield(accumulator)
                    end

                    next_node = nodes.first

                    if next_node.get_type.is_singleton?
                        # This node has a singleton type. We're done with this one.
                        return build_switch_for_args(nodes.drop(1), accumulator + [next_node], &block)
                    else
                        return generate_polymorphic_switch(next_node) do |sing_node|
                            build_switch_for_args(nodes.drop(1), accumulator + [sing_node], &block)
                        end
                    end
                end

                def generate_send_for_singleton(node, singleton_recv, return_type)
                    recv_type = singleton_recv.get_type.singleton_type

                    if RubyIntegration.has_implementation?(recv_type, node.selector)
                        # Some implementations accept only singleton-typed arguments
                        if RubyIntegration.expect_singleton_args?(recv_type, node.selector)
                            # Generate additional switch statements (one per non-sing. arg.).
                            # Go through all possible combinations of types (for arguments).
                            result_identifier = "_polytemp_result_#{temp_identifier_id}"
                            declare_result_var = "#{return_type.to_c_type} #{result_identifier};\n"

                            case_stmts = build_switch_for_args(node.arguments) do |all_sing_args|
                                # TODO: Do we really have to redo type inference here?
                                all_sing_arg_types = all_sing_args.map do |arg|
                                    arg.get_type
                                end

                                this_return_type = RubyIntegration.get_return_type(
                                    singleton_recv.get_type.singleton_type, 
                                    node.selector, 
                                    *all_sing_arg_types, 
                                    args_ast: node.arguments, 
                                    block_ast: node.block_argument)

                                impl = RubyIntegration.get_implementation(
                                    singleton_recv,
                                    node.selector, 
                                    all_sing_args, 
                                    translator,
                                    this_return_type)

                                if this_return_type.is_singleton? and
                                    !return_type.is_singleton?

                                    impl = wrap_in_union_type(
                                        impl, 
                                        this_return_type.singleton_type)
                                end

                                "#{result_identifier} = #{impl};"
                            end

                            return "(" + wrap_in_c_block(
                                declare_result_var + 
                                wrap_in_c_block(case_stmts) + 
                                    result_identifier + ";")[0..-2] + ")"
                        else
                            # The easy case: Anything is fine (but might fail in ruby_integration)
                            return RubyIntegration.get_implementation(
                                singleton_recv,
                                node.selector, 
                                node.arguments, 
                                translator,
                                return_type)
                        end
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
                    return "union_t(#{type.class_id}, union_v_t::from_int(#{str}))"
                elsif type == Types::PrimitiveType::Float
                    return "union_t(#{type.class_id}, union_v_t::from_float(#{str}))"
                elsif type == Types::PrimitiveType::Bool
                    return "union_t(#{type.class_id}, union_v_t::from_bool(#{str}))"
                elsif type == Types::PrimitiveType::Nil
                    return "union_t(#{type.class_id}, union_v_t::from_int(#{str}))"
                elsif type.is_a?(Symbolic::ArrayCommand)
                    return "union_t(#{type.class_id}, union_v_t::from_array_command_t((array_command_t<void> *) #{str}))"
                elsif type.is_a?(Types::LocationAwareFixedSizeArrayType)
                    return "union_t(#{type.class_id}, union_v_t::from_fixed_size_array_t(#{str}))"
                elsif !type.is_a?(Types::UnionType)
                    return "union_t(#{type.class_id}, union_v_t::from_object_id(#{str}))"
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