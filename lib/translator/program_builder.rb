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
                attr_reader :kernel_launchers
                attr_reader :kernels
                attr_reader :root_command

                # An array of structs definitions ([Types::StructType] instances) that should be 
                # generated for this program.
                attr_reader :structs

                def initialize(environment_builder:, root_command:)
                    @kernel_launchers = []
                    @kernels = Set.new([])
                    @environment_builder = environment_builder
                    @root_command = root_command

                    # The collection of structs is a [Set]. Struct types are unique, i.e., there
                    # are never two equal struct types with different object identity.  
                    @structs = Set.new
                end

                def add_kernel_launcher(launcher)
                    @kernel_launchers.push(launcher)
                end

                def assert_ready_to_build
                    if kernel_launchers.size == 0
                        raise "Not ready to build (ProgramBuilder): No kernel launcher defined"
                    end
                end

                # Generates the source code for the CUDA program, compiles it with nvcc and
                # executes the program.
                def execute
                    source = build_program
                    launcher = Launcher.new(
                        source: source,
                        environment_builder: environment_builder,
                        return_type: kernel_launchers.last.kernel_builder.result_type,
                        result_size: kernel_launchers.last.num_threads,
                        root_command: root_command)

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

                    # Generate all struct types
                    for struct_type in structs
                        result = result + struct_type.generate_definition + "\n"
                    end

                    # Build methods, blocks and kernels
                    for launcher in kernel_launchers
                        # Check whether kernel was already build before
                        if kernels.include?(launcher.kernel_builder)
                            next
                        else
                            kernels.add(launcher.kernel_builder)
                        end

                        result = result + launcher.kernel_builder.build_methods
                        result = result + launcher.kernel_builder.build_blocks
                        result = result + launcher.kernel_builder.build_kernel
                    end

                    # Read some fields from last kernel launch configuration
                    final_kernel_result_var = kernel_launchers.last.host_result_var_name
                    if final_kernel_result_var == nil
                        raise "Result variable name of final kernel launcher not set"
                    end

                    final_kernel_result_type = kernel_launchers.last.kernel_builder.result_type.to_c_type

                    # Build kernel invocations
                    program_launch = ""

                    for launcher in kernel_launchers
                        program_launch = program_launch + launcher.build_kernel_launcher
                    end

                    # Free device memory
                    free_device_memory = ""

                    for launcher in kernel_launchers
                        if !launcher.reuse_memory?
                            free_device_memory = free_device_memory + launcher.build_device_memory_free
                        end
                    end

                    # Build program entry point
                    result = result + Translator.read_file(file_name: "entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "result_type" => final_kernel_result_type,
                        "launch_all_kernels" => program_launch,
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
