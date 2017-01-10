module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_identity_command(command)
                Log.info("Translating ArrayIdentityCommand [#{command.unique_id}]")

                super

                # This is a root command, determine grid/block dimensions
                kernel_launcher.configure_grid(command.size, block_size: command.block_size)

                # Add base array to environment
                need_union_type = !command.base_type.is_singleton?
                transformed_base_array = object_tracer.convert_base_array(
                    command.input.first.command, need_union_type)
                environment_builder.add_base_array(command.unique_id, transformed_base_array)

                command_translation = build_command_translation_result(
                    result: "#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[_tid_]",
                    command: command)

                Log.info("DONE translating ArrayIdentityCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
