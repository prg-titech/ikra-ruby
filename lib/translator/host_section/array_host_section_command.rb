require_relative "parallel_section_invocation_visitor"
require_relative "program_builder"
require_relative "ast_translator"
require_relative "../../ast/ssa_generator"

module Ikra
    module Translator
        class HostSectionCommandTranslator < CommandTranslator
            def initialize(root_command:)
                super

                # Use a different program builder
                @program_builder = HostSectionProgramBuilder.new(
                    environment_builder: environment_builder, 
                    root_command: root_command)
            end

            def start_translation
                Log.info("HostSectionCommandTranslator: Starting translation...")

                # Trace all objects
                @object_tracer = TypeInference::ObjectTracer.new(root_command)
                all_objects = object_tracer.trace_all

                # Translate the command (might create additional kernels)
                root_command.accept(self)

                # Add SoA arrays to environment
                object_tracer.register_soa_arrays(environment_builder)
            end

            def visit_array_host_section_command(command)
                Log.info("Translating ArrayHostSectionCommand [#{command.unique_id}]")

                super

                # A host section must be a top-level (root) command. It uses a special
                # [HostSectionProgramBuilder].

                block_def_node = command.block_def_node

                # Cannot use the normal `translate_block` method here, this is special!
                # TODO: There's some duplication here with [BlockTranslator]

                # Build hash of parameter name -> type mappings
                block_parameter_types = {}
                command.block_parameter_names.each_with_index do |name, index|
                    block_parameter_types[name] = command.section_input[index].ikra_type.to_union_type
                end

                parameter_types_string = "[" + block_parameter_types.map do |id, type| "#{id}: #{type}" end.join(", ") + "]"
                Log.info("Translating block with input types #{parameter_types_string}")

                # Add information to block_def_node
                block_def_node.parameters_names_and_types = block_parameter_types

                # Insert return statements (also done by type inference visitor, but we need
                # it now)
                block_def_node.accept(LastStatementReturnsVisitor.new)

                # Insert synthetic __call__ send nodes for return values
                block_def_node.accept(ParallelSectionInvocationVisitor.new)

                # Concert to SSA form
                AST::SSAGenerator.transform_to_ssa!(block_def_node)

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                result_type = type_inference_visitor.process_block(block_def_node)

                for singleton_type in result_type
                    if !singleton_type.is_a?(Types::LocationAwareVariableSizeArrayType)
                        raise AssertionError.new("Return value of host section must be an LocationAwareVariableSizeArrayType. Found a code path with #{singleton_type}.")
                    end
                end

                # C++/CUDA code generation
                ast_translator = HostSectionASTTranslator.new(command_translator: self)

                # Auxiliary methods are instance methods that are called by the host section
                aux_methods = type_inference_visitor.all_methods.map do |method|
                    ast_translator.translate_method(method)
                end

                # Build C++ function
                function_translation = ast_translator.translate_block(block_def_node)

                # Declare local variables
                block_def_node.local_variables_names_and_types.each do |name, type|
                    function_translation.prepend("#{type.to_c_type} #{name};\n")
                end

                mangled_name = "_host_section_#{command.unique_id}_"
                function_parameters = [
                    "#{Constants::ENV_TYPE} *#{Constants::ENV_HOST_IDENTIFIER}",
                    "#{Constants::ENV_TYPE} *#{Constants::ENV_DEVICE_IDENTIFIER}",
                    "#{Constants::PROGRAM_RESULT_TYPE} *#{Constants::PROGRAM_RESULT_IDENTIFIER}"]

                # Define incoming values (parameters). These must all be array commands for now.
                parameter_def = block_parameter_types.map do |name, type|
                    if type.singleton_type.is_a?(Symbolic::ArrayCommand)
                        # Should be initialized with new array command struct
                        "#{type.singleton_type.to_c_type} #{name} = new #{type.singleton_type.to_c_type[0...-2]}();"
                    else
                        "#{type.singleton_type.to_c_type} #{name};"
                    end
                end.join("\n") + "\n"

                translation_result = Translator.read_file(
                    file_name: "host_section_block_function_head.cpp",
                    replacements: { 
                        "name" => mangled_name, 
                        "result_type" => result_type.to_c_type,
                        "parameters" => function_parameters.join(", "),
                        "body" => Translator.wrap_in_c_block(parameter_def + function_translation)})

                program_builder.host_section_source = translation_result

                # Build function invocation
                args = [
                    Constants::ENV_HOST_IDENTIFIER, 
                    Constants::ENV_DEVICE_IDENTIFIER,
                    Constants::PROGRAM_RESULT_IDENTIFIER]

                # Generate code that transfers data back to host. By creating a synthetic send
                # node here, we can let the compiler generate a switch statement if the type of
                # the return value (array) cannot be determined uniquely at compile time.
                host_section_invocation = AST::SourceCodeExprNode.new(
                    code: "#{mangled_name}(#{args.join(", ")})")
                host_section_invocation.get_type.expand(result_type)
                device_to_host_transfer_node = AST::SendNode.new(
                    receiver: host_section_invocation,
                    selector: :__to_host_array__)

                # Type inference is a prerequisite for code generation
                type_inference_visitor.visit_send_node(device_to_host_transfer_node)

                program_builder.host_result_expression = device_to_host_transfer_node.accept(
                    ast_translator.expression_translator)
                program_builder.result_type = device_to_host_transfer_node.get_type

                Log.info("DONE translating ArrayHostSectionCommand [#{command.unique_id}]")

                # This method has no return value (for the moment)
            end
        end
    end
end

require_relative "array_in_host_section_command"
