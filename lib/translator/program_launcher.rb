require "ffi"

module Ikra
    module Translator
        class CommandTranslator
            class ProgramBuilder
                class Launcher
                    class CommandNotifier < Symbolic::Visitor
                        # This visitor executes the post_execute function on every Array Command
                        # That way the responsible command node will recieve the adress of the computed result that will be kept on GPU

                        attr_reader :environment

                        def initialize(environment)
                            @environment = environment
                        end

                        def visit_array_command(command)
                            super(command)
                            command.post_execute(environment)
                        end
                    end

                    class FixedSizeArrayStruct < FFI::Struct
                        layout :content, :pointer,
                            :size, :int32
                    end

                    class KernelResultStruct < FFI::Struct
                        layout :result, FixedSizeArrayStruct,
                            :error_code, :int32,
                            :time_setup_cuda, :uint64,
                            :time_prepare_env, :uint64,
                            :time_kernel, :uint64,
                            :time_free_memory, :uint64
                    end

                    attr_reader :root_command
                    attr_reader :source
                    attr_reader :environment_builder
                    attr_reader :result_type

                    class << self
                        attr_accessor :last_time_setup_cuda
                        attr_accessor :last_time_prepare_env
                        attr_accessor :last_time_kernel
                        attr_accessor :last_time_free_memory
                        attr_accessor :last_time_total_external
                        attr_accessor :last_time_compiler
                        attr_accessor :last_time_read_result_ffi
                    end

                    def initialize(source:, environment_builder:, result_type:, root_command:)
                        @source = source
                        @environment_builder = environment_builder
                        @result_type = result_type
                        @root_command = root_command
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
                        self.class.last_time_compiler = Time.now - time_before

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
                        total_time_external = Time.now - time_before
                        Log.info("Kernel time: #{total_time_external} s")

                        # Update command
                        root_command.accept(CommandNotifier.new(environment_builder.ffi_struct))

                        # Extract error code and return value
                        result_t_struct = KernelResultStruct.new(kernel_result)
                        error_code = result_t_struct[:error_code]

                        # Extract time measurements
                        self.class.last_time_setup_cuda = result_t_struct[:time_setup_cuda] * 0.000001
                        self.class.last_time_prepare_env = result_t_struct[:time_prepare_env] * 0.000001
                        self.class.last_time_kernel = result_t_struct[:time_kernel] * 0.000001
                        self.class.last_time_free_memory = result_t_struct[:time_free_memory] * 0.000001
                        self.class.last_time_total_external = total_time_external

                        if error_code != 0
                            # Kernel failed
                            Errors.raiseCudaError(error_code)
                        end

                        time_before = Time.now

                        # Check type of result: It should be one of `result_type`
                        if result_type.is_singleton?
                           array_type = result_type.singleton_type

                            if !array_type.is_a?(Types::ArrayType)
                                raise "ArrayType expected, but #{array_type} found"
                            end

                            result = result_t_struct[:result][:content]
                            result_size = result_t_struct[:result][:size]
                        else
                            raise NotImplementedError.new
                        end

                        inner_type = array_type.inner_type

                        if inner_type.is_singleton?
                            # Read in entire array
                            if inner_type.singleton_type == Types::PrimitiveType::Int
                                computation_result = result.read_array_of_int(result_size)
                            elsif inner_type.singleton_type == Types::PrimitiveType::Float
                                computation_result = result.read_array_of_float(result_size)
                            elsif inner_type.singleton_type == Types::PrimitiveType::Bool
                                computation_result = result.read_array_of_uint8(result_size).map do |v|
                                    v == 1
                                end
                            elsif inner_type.singleton_type == Types::PrimitiveType::Nil
                                computation_result = [nil] * result_size
                            elsif inner_type.singleton_type.is_a?(Types::ZipStructType)
                                result_struct_type = inner_type.singleton_type.to_ruby_type
                                computation_result = Array.new(result_size) do |index|
                                    result_struct_type.new(result + index * result_struct_type.size)
                                end
                            else
                                raise NotImplementedError.new("Type not implemented")
                            end

                            self.class.last_time_read_result_ffi = Time.now - time_before
                            return computation_result
                        else
                            # Read union type struct
                            # Have to read one by one and assemble object
                            result_values = Array.new(result_size)

                            for index in 0...result_size
                                # TODO: Size of union type (12 bytes) should not be hard-coded here
                                s = Constants::UNION_TYPE_SIZE
                                o = Constants::UNION_TYPE_VALUE_OFFSET
                                next_type = (result + (s * index)).read_int

                                if next_type == Types::PrimitiveType::Int.class_id
                                    result_values[index] = (result + s * index + o).read_int
                                elsif next_type == Types::PrimitiveType::Float.class_id
                                    result_values[index] = (result + s * index + o).read_float
                                elsif next_type == Types::PrimitiveType::Bool.class_id
                                    result_values[index] = (result + s * index + o).read_uint8 == 1
                                elsif next_type == Types::PrimitiveType::Nil.class_id
                                    result_values[index] = nil
                                else
                                    raise NotImplementedError.new("Implement class objs for \##{index}: #{next_type}")
                                end
                            end

                            self.class.last_time_read_result_ffi = Time.now - time_before
                            return result_values
                        end
                    end
                end
            end
        end
    end
end

