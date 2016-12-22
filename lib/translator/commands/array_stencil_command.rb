module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_stencil_command(command)
                Log.info("Translating ArrayStencilCommand [#{command.unique_id}]")

                super

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input_translated = command.input.first.translate_input(
                    command: command,
                    command_translator: self)

                # Count number of parameters
                num_parameters = command.offsets.size

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: input_translated.parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id,
                    pre_execution: input_translated.pre_execution,
                    override_block_parameters: input_translated.override_block_parameters)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # `previous_result` should be an expression returning the array containing the
                # result of the previous computation.
                previous_result = input_translated.command_translation_result.result

                arguments = ["_env_"]

                # Pass values from previous computation that are required by this thread.
                for i in 0...num_parameters
                    arguments.push("#{previous_result}[_tid_ + #{command.offsets[i]}]")
                end

                argument_str = arguments.join(", ")
                stencil_computation = block_translation_result.function_name + "(#{argument_str})"

                temp_var_name = "temp_stencil_#{CommandTranslator.next_unique_id}"

                # The following template checks if there is at least one index out of bounds. If
                # so, the fallback value is used. Otherwise, the block is executed.
                command_execution = Translator.read_file(file_name: "stencil_body.cpp", replacements: {
                    "execution" => input_translated.command_translation_result.execution,
                    "temp_var" => temp_var_name,
                    "result_type" => block_translation_result.result_type.to_c_type,
                    "min_offset" => command.min_offset.to_s,
                    "max_offset" => command.max_offset.to_s,
                    "thread_id" => "_tid_",
                    "input_size" => command.input.first.command.size.to_s,
                    "out_of_bounds_fallback" => command.out_of_range_value.to_s,
                    "stencil_computation" => stencil_computation})

                command_translation = build_command_translation_result(
                    execution: command_execution,
                    result: temp_var_name,
                    return_type: block_translation_result.result_type,
                    keep: command.keep,
                    unique_id: command.unique_id,
                    command: command)

                kernel_launcher.set_result_name(command.unique_id.to_s)

                Log.info("DONE translating ArrayStencilCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
