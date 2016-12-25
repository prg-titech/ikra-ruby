module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_zip_command(command)
                Log.info("Translating ArrayZipCommand [#{command.unique_id}]")
                
                super

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input = translate_entire_input(command)

                # Build Ikra struct type
                zipped_type_singleton = Types::ZipStructType.new(*input.result_type)
                zipped_type = Types::UnionType.new(zipped_type_singleton)

                # Add struct type to program builder, so that we can generate the source code
                # for its definition.
                program_builder.structs.add(zipped_type_singleton)

                command_translation = CommandTranslationResult.new(
                    execution: input.execution,
                    result: zipped_type_singleton.generate_inline_initialization(input.result),
                    result_type: zipped_type)

                Log.info("DONE translating ArrayZipCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
