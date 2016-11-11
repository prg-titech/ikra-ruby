require "ffi"

module Ikra
    module Translator
        class CommandTranslator
            class ProgramBuilder
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

