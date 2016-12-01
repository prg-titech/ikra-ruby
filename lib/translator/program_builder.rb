require "tempfile"
require "set"

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
                attr_reader :kernel_launch_configurations
                attr_reader :kernels

                def initialize(environment_builder:)
                    @kernel_launch_configurations = []
                    @kernels = Set.new([])
                    @environment_builder = environment_builder
                end

                def add_kernel_launcher(launch_configuration)
                    @kernel_launch_configurations.push(launch_configuration)
                end

                def assert_ready_to_build
                    if kernel_launch_configurations.size == 0
                        raise "Not ready to build (ProgramBuilder): No kernel launch configurations defined"
                    end
                end

                # Generates the source code for the CUDA program, compiles it with nvcc and
                # executes the program.
                def execute
                    source = build_program
                    launcher = Launcher.new(
                        source: source,
                        environment_builder: environment_builder,
                        return_type: kernel_launch_configurations.last.kernel_builder.result_type,
                        result_size: kernel_launch_configurations.last.num_threads)

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
                    for configuration in kernel_launch_configurations
                        # Check whether kernel was already build before
                        if kernels.include?(configuration.kernel_builder)
                            next
                        else
                            kernels.add(configuration.kernel_builder)
                        end

                        result = result + configuration.kernel_builder.build_methods
                        result = result + configuration.kernel_builder.build_blocks
                        result = result + configuration.kernel_builder.build_kernel
                    end

                    # Read some fields from last kernel launch configuration
                    final_kernel_result_var = kernel_launch_configurations.last.host_result_var_name
                    if final_kernel_result_var == nil
                        raise "Result variable name of final kernel launch configuration not set"
                    end

                    final_kernel_result_type = kernel_launch_configurations.last.kernel_builder.result_type.to_c_type

                    # Build kernel invocations
                    kernel_launchers = ""

                    for configuration in kernel_launch_configurations
                        kernel_launchers = kernel_launchers + configuration.build_kernel_launcher
                    end

                    # Build program entry point
                    result = result + Translator.read_file(file_name: "entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "result_type" => final_kernel_result_type,
                        "launch_all_kernels" => kernel_launchers,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_var_name" => final_kernel_result_var})

                    return result
                end
            end
        end
    end
end

require_relative "program_launcher"
