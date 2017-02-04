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

                # Generates the source code for the CUDA program, compiles it with nvcc and
                # executes the program.
                def execute
                    source = build_program

                    launcher = Launcher.new(
                        source: source,
                        environment_builder: environment_builder,
                        result_type: result_type,
                        root_command: root_command)

                    launcher.compile
                    return launcher.execute
                end

                protected

                def assert_ready_to_build
                    if kernel_launchers.size == 0
                        raise "Not ready to build (ProgramBuilder): No kernel launcher defined"
                    end
                end

                # Build header of CUDA source code file
                def build_header
                    return Translator.read_file(file_name: "header.cpp")
                end

                # Build environment struct definition
                def build_environment_struct
                    return environment_builder.build_environment_struct
                end

                # Generate all struct types
                def build_struct_types
                    return structs.map do |struct_type|
                        struct_type.generate_definition
                    end.join("\n")
                end

                # Build methods, blocks and kernels
                def build_kernels
                    result = ""

                    for launcher in kernel_launchers
                        for builder in launcher.kernel_builders
                            # Check whether kernel was already build before
                            if kernels.include?(builder)
                                next
                            else
                                kernels.add(builder)
                            end

                            result = result + builder.build_methods
                            result = result + builder.build_blocks
                            result = result + builder.build_kernel
                        end
                    end

                    return result
                end

                # Build kernel invocations
                def build_kernel_launchers
                    return kernel_launchers.map do |launcher|
                        launcher.build_kernel_launcher
                    end.join("")
                end

                def host_result_expression
                    # Read some fields from last kernel launch configuration
                    result_device_ptr = kernel_launchers.last.kernel_result_var_name
                    result_c_type = kernel_launchers.last.result_type.to_c_type
                    result_size = root_command.size

                    if result_device_ptr == nil
                        raise "Result variable name of final kernel launcher not set"
                    end

                    # Build result values: `fixed_size_array_t` struct. This struct contains a
                    # pointer to the result array and stores the size of the result.
                    result_device_fixed_array_t = "fixed_size_array_t<#{result_c_type}>(#{result_device_ptr}, #{result_size})"

                    return Translator.read_file(file_name: "memcpy_device_to_host_expr.cpp", replacements: {
                        "type" => result_c_type,
                        "device_array" => result_device_fixed_array_t})
                end

                # Returns the result type of this program. The result type must always be a
                # union type that includes a[Types::LocationAwareFixedSizeArrayType] object, 
                # because this way we can support return types where the inner type of an array
                # is unknown at compile time.
                def result_type
                    return Types::LocationAwareFixedSizeArrayType.new(
                        kernel_launchers.last.result_type,
                        location: :host).to_union_type
                end

                # Free device memory
                def build_memory_free
                    result = ""

                    for launcher in kernel_launchers
                        if !launcher.reuse_memory?
                            result = result + launcher.build_device_memory_free
                        end
                    end

                    return result
                end

                # Build the struct type for `result_t`.
                def build_header_structs
                    header_structs = Translator.read_file(file_name: "header_structs.cpp",
                        replacements: {"result_type" => result_type.to_c_type})
                end

                # Builds the CUDA program. Returns the source code string.
                def build_program
                    assert_ready_to_build

                    result = build_header + build_struct_types + build_header_structs + build_environment_struct +  build_kernels

                    # Build program entry point
                    return result + Translator.read_file(file_name: "entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "launch_all_kernels" => build_kernel_launchers,
                        "free_device_memory" => build_memory_free,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_array" => host_result_expression})
                end
            end
        end
    end
end

require_relative "program_launcher"
