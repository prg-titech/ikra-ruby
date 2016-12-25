module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_combine_command(command)
                Log.info("Translating ArrayCombineCommand [#{command.unique_id}]")

                super

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input = translate_entire_input(command)

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id,
                    entire_input_translation: input)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Build command invocation string
                result = block_translation_result.function_name + "(" + 
                    (["_env_"] + input.result).join(", ") + ")"

                command_translation = build_command_translation_result(
                    execution: input.execution,
                    result: result,
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
