module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_combine_command(command)
                Log.info("Translating ArrayCombineCommand [#{command.unique_id}]")

                super

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input_translated = command.input.each_with_index.map do |input, index|
                    input.translate_input(
                        command: command,
                        command_translator: self,
                        start_eat_params_offset: index)
                end

                # Get all parameters
                block_parameters = input_translated.map do |input|
                    input.parameters
                end.reduce(:+)

                # Get all pre-execution statements
                pre_execution = input_translated.map do |input|
                    input.pre_execution
                end.reduce(:+)

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: block_parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    pre_execution: pre_execution,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Build command invocation string
                command_args = (["_env_"] + input_translated.map do |input|
                    input.command_translation_result.result
                end).join(", ")


                input_execution = input_translated.map do |input|
                    input.command_translation_result.execution
                end.join("\n\n")

                command_translation = build_command_translation_result(
                    execution: input_execution,
                    result: block_translation_result.function_name + "(" + command_args + ")",
                    result_type: block_translation_result.result_type,
                    keep: command.keep,
                    unique_id: command.unique_id,
                    command: command)

                kernel_launcher.set_result_name(command.unique_id.to_s)

                Log.info("DONE translating ArrayCombineCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
