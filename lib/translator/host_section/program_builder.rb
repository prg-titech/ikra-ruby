require "set"

require_relative "../program_builder"

module Ikra
    module Translator
        class CommandTranslator
            class HostSectionProgramBuilder < ProgramBuilder
                # A host C++ function containing the source code of the host section.
                attr_accessor :host_section_source

                # The type of the result (not an array type, just the inner type).
                attr_accessor :result_type

                # An expression that returns the final result, as an `variable_size_array_t` object
                # pointing to an array in the host memory.
                attr_accessor :host_result_expression

                def initialize(environment_builder:, root_command:)
                    super

                    @kernel_builders = Set.new
                end

                def assert_ready_to_build
                    if host_section_source == nil
                        raise AssertionError.new("Not ready to build (HostSectionProgramBuilder): No host section source code defined")
                    end

                    if result_type == nil
                        raise AssertionError.new("Not ready to build (HostSectionProgramBuilder): No result type defined")
                    end

                    if host_result_expression == nil
                        raise AssertionError.new("Not ready to build (HostSectionProgramBuilder): No host result expression defined")
                    end
                end

                def clear_kernel_launchers
                    @kernel_launchers.clear
                end

                def add_kernel_launcher(launcher)
                    super

                    # Let's keep track of kernels here by ourselves
                    @kernel_builders.merge(launcher.kernel_builders)
                end

                def all_kernel_builders
                    return @kernel_builders
                end

                def prepare_additional_args_for_launch(command)
                    kernel_launchers.each do |launcher|
                        launcher.prepare_additional_args_for_launch(command)
                    end
                end

                # Builds the CUDA program. Returns the source code string.
                def build_program
                    assert_ready_to_build

                    result = build_header + build_struct_types + build_header_structs + 
                        build_array_command_struct_types + build_environment_struct + 
                        build_kernels + host_section_source

                    # Build program entry point
                    return result + Translator.read_file(file_name: "host_section_entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_array" => host_result_expression})
                end
            end
        end
    end
end