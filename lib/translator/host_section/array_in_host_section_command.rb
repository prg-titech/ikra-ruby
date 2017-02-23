module Ikra
    module Translator
        class HostSectionCommandTranslator < CommandTranslator
            def visit_array_in_host_section_command(command)
                Log.info("Translating ArrayInHostSectionCommand [#{command.unique_id}]")

                super

                # This is a root command, determine grid/block dimensions
                kernel_launcher.configure_grid(command.size, block_size: command.block_size)

                array_input_id = "_array_#{self.class.next_unique_id}_"
                kernel_builder.add_additional_parameters("#{command.base_type.to_c_type} *#{array_input_id}")

                # Add placeholder for argument (input array). This should be done here to preserve
                # the order or arguments.
                kernel_launcher.add_additional_arguments(proc do |cmd|
                    # `cmd` is a reference to the command being launched (which might be merged
                    # with other commands). Based on that information, we can generate an 
                    # expression that returns the input array.
                    arg = Translator::KernelLaunchArgumentGenerator.generate_arg(
                        command, cmd, "cmd")

                    if arg == nil
                        raise AssertionError.new("Argument not found: Trying to launch command #{cmd.unique_id}, looking for result of command #{command.unique_id}")
                    end

                    arg
                end)

                command_translation = build_command_translation_result(
                    result: "#{array_input_id}[_tid_]",
                    command: command)

                Log.info("DONE translating ArrayInHostSectionCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
