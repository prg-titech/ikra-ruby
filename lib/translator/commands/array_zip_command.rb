module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_zip_command(command)
                Log.info("Translating ArrayZipCommand [#{command.unique_id}]")
                
                super

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input_translated = command.input.each_with_index.map do |input, index|
                    input.translate_input(
                        command: command,
                        command_translator: self,
                        start_eat_params_offset: index)
                end

                # Execute input
                input_execution = input_translated.map do |input|
                    input.command_translation_result.execution
                end.join("\n\n")

                # Get result of execution
                input_result = input_translated.map do |input|
                    input.command_translation_result.result
                end

                # Build Ikra struct type
                zipped_type_singleton = Types::ZipStructType.new(*(
                        input_translated.map do |input| 
                        input.return_type
                    end))

                zipped_type = Types::UnionType.new(zipped_type_singleton)

                # Add struct type to program builder, so that we can generate the source code
                # for its definition.
                program_builder.structs.add(zipped_type_singleton)

                command_translation = CommandTranslationResult.new(
                    execution: input_execution,
                    result: zipped_type_singleton.generate_inline_initialization(input_result),
                    return_type: zipped_type)

                Log.info("DONE translating ArrayZipCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
