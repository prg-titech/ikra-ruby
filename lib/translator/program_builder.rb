require "tempfile"
require "ffi"

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

                    # Build program entry point
                    result = result + Translator.read_file(file_name: "entry_point.cpp", replacements: {
                        "prepare_environment" => environment_builder.build_environment_variable,
                        "result_type" => final_kernel_result_type,
                        "launch_all_kernels" => kernel_launchers,
                        "host_env_var_name" => Constants::ENV_HOST_IDENTIFIER,
                        "host_result_var_name" => final_kernel_result_var})

                    return result
                end

                class Launcher
                    class KernelResultStruct < FFI::Struct
                        layout :result, :pointer,
                            :error_code, :int32
                    end

                    attr_reader :source
                    attr_reader :environment_builder
                    attr_reader :return_type
                    attr_reader :result_size

                    def initialize(source:, environment_builder:, return_type:, result_size:)
                        @source = source
                        @environment_builder = environment_builder
                        @return_type = return_type
                        @result_size = result_size
                    end

                    def compile
                        # Generate debug output with line numbers
                        line_no_digits = Math.log(source.lines.count, 10).ceil
                        source_with_line_numbers = source.lines.each_with_index.map do |line, num| 
                            "[#{(num + 1).to_s.rjust(line_no_digits, "0")}] #{line}" 
                        end.join("")

                        Log.info("Generated source code:\n#{source_with_line_numbers}")

                        # Write source code to temporary file
                        file = Tempfile.new(["ikra_kernel", ".cu"])
                        file.write(source)
                        file.close

                        # Write to codegen_expect
                        if Configuration.codegen_expect_file_name != nil
                            expect_file = File.new(Configuration.codegen_expect_file_name, "w+")
                            expect_file.write(source)
                            expect_file.close
                        end

                        # Run compiler
                        @so_filename = "#{file.path}.#{Configuration.so_suffix}"
                        nvcc_command = Configuration.nvcc_invocation_string(
                            file.path, @so_filename)

                        Log.info("Compiling kernel: #{nvcc_command}")
                        time_before = Time.now
                        compile_status = %x(#{nvcc_command})
                        Log.info("Done, took #{Time.now - time_before} s")

                        if $? != 0
                            Log.fatal("nvcc failed: #{compile_status}")
                            raise "nvcc failed: #{compile_status}"
                        else
                            Log.info("nvcc successful: #{compile_status}")
                        end
                    end

                    # Attaches the compiled shared library via Ruby FFI and invokes the kernel.
                    def execute
                        if !File.exist?(@so_filename)
                            compile
                        end

                        time_before = Time.now
                        ffi_interface = Module.new
                        ffi_interface.extend(FFI::Library)
                        ffi_interface.ffi_lib(@so_filename)
                        ffi_interface.attach_function(:launch_kernel, [:pointer], :pointer)
                        environment_object = environment_builder.build_ffi_object
                        Log.info("FFI transfer time: #{Time.now - time_before} s")

                        time_before = Time.now
                        kernel_result = ffi_interface.launch_kernel(environment_object)
                        Log.info("Kernel time: #{Time.now - time_before} s")

                        # Extract error code and return value
                        result_t_struct = KernelResultStruct.new(kernel_result)
                        error_code = result_t_struct[:error_code]

                        if error_code != 0
                            # Kernel failed
                            Errors.raiseCudaError(error_code)
                        end

                        result = result_t_struct[:result]

                        if return_type.is_singleton?
                            # Read in entire array
                            if return_type.singleton_type == Types::PrimitiveType::Int
                                return result.read_array_of_int(result_size)
                            elsif return_type.singleton_type == Types::PrimitiveType::Float
                                return result.read_array_of_float(result_size)
                            elsif return_type.singleton_type == Types::PrimitiveType::Bool
                                return result.read_array_of_uint8(result_size).map do |v|
                                    v == 1
                                end
                            elsif return_type.singleton_type == Types::PrimitiveType::Nil
                                return [nil] * result_size
                            else
                                raise NotImplementedError.new("Type not implemented")
                            end
                        else
                            # Read union type struct
                            # Have to read one by one and assemble object
                            result_values = Array.new(result_size)

                            for index in 0...result_size
                                next_type = (result + (8 * index)).read_int

                                if next_type == Types::PrimitiveType::Int.class_id
                                    result_values[index] = (result + 8 * index + 4).read_int
                                elsif next_type == Types::PrimitiveType::Float.class_id
                                    result_values[index] = (result + 8 * index + 4).read_float
                                elsif next_type == Types::PrimitiveType::Bool.class_id
                                    result_values[index] = (result + 8 * index + 4).read_uint8 == 1
                                elsif next_type == Types::PrimitiveType::Nil.class_id
                                    result_values[index] = nil
                                else
                                    raise NotImplementedError.new("Implement class objs for \##{index}: #{next_type}")
                                end
                            end

                            return result_values
                        end
                    end
                end
            end
        end
    end
end