require_relative "../program_builder"

module Ikra
    module Translator
        class CommandTranslator
            class HostSectionProgramBuilder < ProgramBuilder
                # A host C++ function containing the source code of the host section.
                attr_accessor :host_section_source

                # A C++ statement that calls the host section function with the correct arguments.
                attr_accessor :host_section_invocation

                # A [Variable] object containing the name and type of a (local) variable containing
                # the result of the host section invocation. This should be device pointer.
                attr_accessor :final_result_variable

                # The size of the result of this host section.
                # TODO: This should be variable. In general, not known at compile time.
                attr_accessor :final_result_size

                def assert_ready_to_build
                    if host_section_source == nil
                        raise "Not ready to build (HostSectionProgramBuilder): No host section source code defined"
                    end

                    if host_section_invocation == nil
                        raise "Not ready to build (HostSectionProgramBuilder): No host section invocation defined"
                    end

                    if final_result_variable == nil
                        raise "Not ready to build (HostSectionProgramBuilder): No final result variable defined"
                    end
                end

                # Builds the CUDA program. Returns the source code string.
                def build_program
                    assert_ready_to_build

                    result = build_header + build_struct_types + build_header_structs + build_environment_struct + build_kernels + host_section_source

                    result_type = final_result_variable.type.to_c_type
                    transfer_result = Translator.read_file(file_name: "host_section_malloc_memcpy_device_to_host.cpp", replacements: {
                        "type" => result_type,
                        "bytes" => "#{final_result_size} * sizeof(#{result_type})"})

                    # Build program entry point
                    return result + Translator.read_file(file_name: "host_section_entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "result_type" => result_type,
                        "launch_all_kernels" => host_section_invocation,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_var_name" => final_result_variable.name,
                        "copy_back_to_host" => transfer_result})
                end
            end
        end
    end
end