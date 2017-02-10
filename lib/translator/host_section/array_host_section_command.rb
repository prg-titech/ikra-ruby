require_relative "parallel_section_invocation_visitor"
require_relative "program_builder"
require_relative "ast_translator"

module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_host_section_command(command)
                Log.info("Translating ArrayHostSectionCommand [#{command.unique_id}]")

                super

                # A host section must be a top-level (root) command. It uses a special
                # [HostSectionProgramBuilder].

                # Translate input (dependent computations)
                # for input_command in command.section_input
                #     push_kernel_launcher
                #     command_translation_result = input_command.accept(self)
                #     result_var = kernel_launcher.kernel_result_var_name
                #     pop_kernel_launcher(command_translation_result)
                # end
                # NOT REQUIRED, BECAUSE INPUT IS FUSED WHEN USED

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

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                result_type = type_inference_visitor.process_block(block_def_node)

                if !result_type.singleton_type.is_a?(Symbolic::ArrayCommand)
                    raise "Return value of host section must be an ArrayCommand"
                end

                # Insert synthetic __call__ send nodes
                block_def_node.accept(ParallelSectionInvocationVisitor.new)

                # C++/CUDA code generation
                ast_translator = HostSectionASTTranslator.new(command_translator: self)

                # Auxiliary methods are instance methods that are called by the host section
                aux_methods = type_inference_visitor.all_methods.map do |method|
                    ast_translator.translate_method(method)
                end

                # Build C++ function
                mangled_name = "_host_section_#{command.unique_id}_"
                function_parameters = [
                    "#{Constants::ENV_TYPE} *#{Constants::ENV_HOST_IDENTIFIER}",
                    "#{Constants::ENV_TYPE} *#{Constants::ENV_DEVICE_IDENTIFIER}",
                    "#{Constants::PROGRAM_RESULT_TYPE} *#{Constants::PROGRAM_RESULT_IDENTIFIER}"]

                function_head = Translator.read_file(
                    file_name: "host_section_block_function_head.cpp",
                    replacements: { 
                        "name" => mangled_name, 
                        "result_type" => result_type.to_c_type,
                        "parameters" => function_parameters.join(", ")})

                function_translation = ast_translator.translate_block(block_def_node)
                
                # Declare local variables
                block_def_node.local_variables_names_and_types.each do |name, type|
                    function_translation.prepend("#{type.to_c_type} #{name};\n")
                end

                translation_result = function_head + 
                    Translator.wrap_in_c_block(function_translation)

                program_builder.host_section_source = translation_result

                # Build function invocation
                args = [
                    Constants::ENV_HOST_IDENTIFIER, 
                    Constants::ENV_DEVICE_IDENTIFIER,
                    Constants::PROGRAM_RESULT_IDENTIFIER]

                program_builder.host_section_invocation = 
                    "#{Constants::PROGRAM_RESULT_IDENTIFIER}->result = #{mangled_name}(#{args.join(", ")})->result;"
                program_builder.final_result_variable = Variable.new(
                    name: "_host_result_",
                    # Retrieve result type from ArrayCommand
                    type: result_type.singleton_type.result_type)
                program_builder.final_result_size = result_type.singleton_type.size

                Log.info("DONE translating ArrayHostSectionCommand [#{command.unique_id}]")

                # This is not an ordinary command, because it is not executed on the device.
                return HostSectionCommandTranslationResult.new
            end

            # This class is just used as a maker to avoid passing `nil`.
            class HostSectionCommandTranslationResult

            end
        end
    end
end
