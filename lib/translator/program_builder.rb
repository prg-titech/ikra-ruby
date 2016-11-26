require "tempfile"

module Ikra
    module Translator
        class CommandTranslator

            # Builds the entire CUDA program. A CUDA program may consist of multiple kernels, but
            # has at least one kernel. The generated code performs the following steps:
            #
            # 1. Insert header of CUDA file.
            # 2. For every kernel: Build all methods, blocks, and kernels.
            # 3. Build the program entry point (including kernel launchers).
            class ProgramBuilder
                attr_reader :environment_builder
                attr_reader :kernels

                def initialize(environment_builder:)
                    @kernels = []
                    @environment_builder = environment_builder
                end

                def add_kernel(kernel)
                    @kernels.push(kernel)
                end

                def assert_ready_to_build
                    if kernels.size == 0
                        raise "Not ready to build (ProgramBuilder): No kernels defined"
                    end
                end

                # Generates the source code for the CUDA program, compiles it with nvcc and
                # executes the program.
                def execute
                    source = build_program
                    launcher = Launcher.new(
                        source: source,
                        environment_builder: environment_builder,
                        return_type: kernels.last.result_type,
                        result_size: kernels.last.num_threads)

                    launcher.compile
                    return launcher.execute
                end

                # Builds the CUDA program. Returns the source code string.
                def build_program
                    assert_ready_to_build

                    # Build header of CUDA source code file
                    result = Translator.read_file(file_name: "header.cpp")

                    # Build environment struct definition
                    result = result + environment_builder.build_environment_struct

                    # Build methods, blocks and kernels
                    for kernel_builder in kernels
                        result = result + kernel_builder.build_methods
                        result = result + kernel_builder.build_blocks
                        result = result + kernel_builder.build_kernel
                    end

                    # Read some fields from last kernel
                    final_kernel_result_var = kernels.last.host_result_var_name
                    if final_kernel_result_var == nil
                        raise "Result variable name of final kernel not set"
                    end

                    final_kernel_result_type = kernels.last.result_type.to_c_type

                    # Build kernel invocations
                    kernel_launchers = ""

                    for kernel_builder in kernels
                        kernel_launchers = kernel_launchers + kernel_builder.build_kernel_lauchner
                    end

                    # Free device memory
                    free_device_memory = ""

                    for kernel_builder in kernels
                        free_device_memory = free_device_memory + kernel_builder.build_device_memory_free
                    end

                    # Build program entry point
                    result = result + Translator.read_file(file_name: "entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "result_type" => final_kernel_result_type,
                        "launch_all_kernels" => kernel_launchers,
                        "free_device_memory" => free_device_memory,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_var_name" => final_kernel_result_var})

                    return result
                end
            end
        end
    end
end

require_relative "program_launcher"
