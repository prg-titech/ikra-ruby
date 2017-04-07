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

                # An array of array command structs.
                attr_reader :array_command_structs

                def initialize(environment_builder:, root_command:)
                    @kernel_launchers = []
                    @kernels = Set.new([])
                    @environment_builder = environment_builder
                    @root_command = root_command
                    @is_compiled = false

                    # The collection of structs is a [Set]. Struct types are unique, i.e., there
                    # are never two equal struct types with different object identity.  
                    @structs = Set.new
                    @array_command_structs = Set.new

                    # Make sure we don't update old IDs in commands multiple times
                    @already_updated_ids = false
                end

                # Reinitializes this builder so that it can be used with a different command that
                # has the exact same source code.
                def reinitialize(root_command:)
                    # Another problem: The environment builder uses unique_id and now we have
                    # different IDs because of new array commands.
                    update_command_env(@root_command, root_command)
                    @already_updated_ids = true

                    # TODO: Update environment_builder. It may have different bindings now.
                    @root_command = root_command
                end

                def update_command_env(old_cmd, new_cmd)
                    if !@already_updated_ids
                        new_cmd.old_unique_id = old_cmd.unique_id
                    else
                        new_cmd.old_unique_id = old_cmd.old_unique_id
                    end

                    if new_cmd.keep && new_cmd.has_previous_result?
                        if !@already_updated_ids
                            environment_builder.add_previous_result(
                                old_cmd.unique_id, new_cmd.gpu_result_pointer)
                        else
                            environment_builder.add_previous_result(
                                old_cmd.old_unique_id, new_cmd.gpu_result_pointer)
                        end
                    end

                    for i in 0...(new_cmd.input.size)
                        next_new_cmd = new_cmd.input[i].command
                        next_old_cmd = old_cmd.input[i].command

                        if next_new_cmd.is_a?(Symbolic::ArrayCommand)
                            update_command_env(next_old_cmd, next_new_cmd)
                        end
                    end
                end

                def is_compiled?
                    return @is_compiled
                end

                def add_array_command_struct(*structs)
                    for struct in structs
                        array_command_structs.add(struct)
                    end
                end

                def add_kernel_launcher(launcher)
                    @kernel_launchers.push(launcher)
                end

                # Generates the source code for the CUDA program, compiles it with nvcc and
                # executes the program.
                def execute
                    source = build_program

                    if !is_compiled?
                        @launcher = Launcher.new(
                            source: source,
                            environment_builder: environment_builder,
                            result_type: result_type,
                            root_command: root_command)

                        @launcher.compile
                        @is_compiled = true
                    else
                        # TODO: Have to update the values in the environment builder!
                        @launcher.reinitialize(
                            environment_builder: environment_builder,
                            root_command: root_command)
                    end

                    return @launcher.execute
                end

                # Build kernel invocations
                def build_kernel_launchers
                    return kernel_launchers.map do |launcher|
                        launcher.build_kernel_launcher
                    end.join("")
                end

                protected

                def assert_ready_to_build
                    if kernel_launchers.size == 0
                        raise AssertionError.new(
                            "Not ready to build (ProgramBuilder): No kernel launcher defined")
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

                # Generate all struct types (except for array command struct types).
                def build_struct_types
                    return structs.map do |struct_type|
                            struct_type.generate_definition
                        end.join("\n") + "\n"
                end

                def build_array_command_struct_types
                    return array_command_structs.to_a.join("\n") + "\n"
                end

                def all_kernel_builders
                    return kernel_launchers.map do |launcher|
                        launcher.kernel_builders
                    end.flatten
                end

                # Build methods, blocks and kernels
                def build_kernels
                    result = ""

                    for builder in all_kernel_builders
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

                    return result
                end

                def host_result_expression
                    # Read some fields from last kernel launch configuration
                    result_device_ptr = kernel_launchers.last.kernel_result_var_name
                    result_c_type = kernel_launchers.last.result_type.to_c_type
                    result_size = root_command.size

                    if result_device_ptr == nil
                        raise AssertionError.new(
                            "Result variable name of final kernel launcher not set")
                    end

                    # Build result values: `variable_size_array_t` struct. This struct contains a
                    # pointer to the result array and stores the size of the result.
                    result_device_variable_array_t = "variable_size_array_t((void *) #{result_device_ptr}, #{result_size})"

                    return Translator.read_file(file_name: "memcpy_device_to_host_expr.cpp", replacements: {
                        "type" => result_c_type,
                        "device_array" => result_device_variable_array_t})
                end

                # Returns the result type of this program. The result type must always be a
                # union type that includes a [Types::LocationAwareArrayType] object, 
                # because this way we can support return types where the inner type of an array
                # is unknown at compile time.
                def result_type
                    return Types::LocationAwareVariableSizeArrayType.new(
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

                    result = build_header + build_struct_types + build_header_structs + 
                        build_array_command_struct_types + build_environment_struct + 
                        build_kernels

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
