require "tempfile"
require "ffi"
require_relative "translator"
require_relative "block_translator"
require_relative "cuda_errors"
require_relative "../config/os_configuration"
require_relative "../symbolic/symbolic"
require_relative "../symbolic/visitor"
require_relative "../types/types"
require_relative "../config/configuration"

module Ikra
    module Translator

        # Result of translating a {Ikra::Symbolic::ArrayCommand}.
        class CommandTranslationResult

            class KernelResultStruct < FFI::Struct
                layout :result, :pointer,
                    :error_code, :int32
            end

            # @return [EnvironmentBuilder] instance that generates the struct containing accessed 
            # lexical variables.
            attr_accessor :environment_builder

            # @return [String] The currently generated source code.
            attr_accessor :generated_source

            # @return [String] Source code used for invoking the block function.
            attr_accessor :invocation

            # @return [Fixnum] The number of elements in the base array.
            attr_accessor :size

            # @return [Types::UnionType] The return type of the block.
            attr_accessor :return_type

            # @return [Fixnum] The size of a block (1D).
            attr_accessor :block_size

            def initialize(environment_builder)
                @environment_builder = environment_builder
                @generated_source = ""
                @invocation = "NULL"
                @return_type = Types::UnionType.new
                @size = 0

                @so_filename = ""                                       # [String] file name of shared library containing CUDA kernel
            end

            def result_size
                @size
            end

            # Compiles CUDA source code and generates a shared library.
            def compile
                grid_dim = [size.fdiv(@block_size).ceil, 1].max
                block_dim = size >= @block_size ? @block_size : size

                # Prepare file replacements
                file_replacements = {}                                  # [Hash{String => String}] contains strings that should be replaced when reading a file 
                file_replacements["grid_dim[0]"] = "#{grid_dim}"
                file_replacements["grid_dim[1]"] = "1"
                file_replacements["grid_dim[2]"] = "1"
                file_replacements["block_dim[0]"] = "#{block_dim}"
                file_replacements["block_dim[1]"] = "1"
                file_replacements["block_dim[2]"] = "1"
                file_replacements["result_type"] = @return_type.to_c_type
                file_replacements["result_size"] = "#{result_size}"
                file_replacements["block_invocation"] = @invocation
                file_replacements["env_identifier"] = Constants::ENV_IDENTIFIER
                file_replacements["copy_env"] = @environment_builder.device_struct_allocation
                file_replacements["dev_env"] = Constants::ENV_DEVICE_IDENTIFIER
                file_replacements["host_env"] = Constants::ENV_HOST_IDENTIFIER

                # Generate source code
                source = Translator.read_file(file_name: "header.cpp", replacements: file_replacements) +
                    @environment_builder.build_struct_definition + 
                    @generated_source +
                    Translator.read_file(file_name: "kernel.cpp", replacements: file_replacements) + 
                    Translator.read_file(file_name: "kernel_launcher.cpp", replacements: file_replacements)

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
                nvcc_command = Configuration.nvcc_invocation_string(file.path, @so_filename)

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

            # Attaches a the compiled shared library via Ruby FFI and invokes the kernel.
            def execute
                if !File.exist?(@so_filename)
                    compile
                end

                time_before = Time.now
                ffi_interface = Module.new
                ffi_interface.extend(FFI::Library)
                ffi_interface.ffi_lib(@so_filename)
                ffi_interface.attach_function(:launch_kernel, [:pointer], :pointer)
                environment_object = @environment_builder.build_ffi_object
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

        # A visitor traversing the tree (currently list) of symbolic array commands. Every command is converted into a {CommandTranslationResult} and possibly merged with the result of dependent (previous) results. This is how kernel fusion is implemented.
        class ArrayCommandVisitor < Symbolic::Visitor
            
            def initialize(environment_builder)
                @environment_builder = environment_builder
            end

            def visit_array_new_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new(@environment_builder)

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    # only one block parameter (int)
                    block_parameter_types: {command.block_parameter_names.first => Types::UnionType.create_int},
                    environment_builder: @environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                command_translation_result.generated_source = block_translation_result.generated_source

                tid = "threadIdx.x + blockIdx.x * blockDim.x"
                command_translation_result.invocation = "#{block_translation_result.function_name}(#{Constants::ENV_IDENTIFIER}, #{tid})"
                command_translation_result.size = command.size
                command_translation_result.return_type = block_translation_result.result_type
                command_translation_result.block_size = command.block_size

                command_translation_result
            end

            def visit_array_identity_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new(@environment_builder)

                # no source code generation

                if Configuration::JOB_REORDERING
                    reordering_array = command.target.each_with_index.sort do |a, b|
                        a.first.class.object_id <=> b.first.class.object_id
                    end.map(&:last)

                    # Generate debug output
                    dbg_elements = []
                    dbg_last = command.target[reordering_array[0]].class
                    dbg_counter = 1

                    for idx in 1..(command.target.size - 1)
                        dbg_next = command.target[reordering_array[idx]].class

                        if dbg_next == dbg_last
                            dbg_counter += 1
                        else
                            dbg_elements.push("#{dbg_last.to_s} (#{dbg_counter})")
                            dbg_last = dbg_next
                            dbg_counter = 1
                        end
                    end
                    dbg_elements.push("#{dbg_last.to_s} (#{dbg_counter})")

                    Log.info("Generated job reordering array, resulting in: [#{dbg_elements.join(", ")}]")

                    reordering_array_name = @environment_builder.add_base_array("#{command.unique_id}j", reordering_array)
                    command_translation_result.invocation = "#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[#{Constants::ENV_IDENTIFIER}->#{reordering_array_name}[threadIdx.x + blockIdx.x * blockDim.x]]"
                else
                    command_translation_result.invocation = "#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[threadIdx.x + blockIdx.x * blockDim.x]"
                end
                
                command_translation_result.size = command.size
                command_translation_result.return_type = command.base_type
                command_translation_result.block_size = command.block_size

                command_translation_result
            end

            def visit_array_map_command(command)
                # visit target (dependent) command
                dependent_result = super

                command_translation_result = CommandTranslationResult.new(@environment_builder)

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameter_types: {command.block_parameter_names.first => dependent_result.return_type},
                    environment_builder: @environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                command_translation_result.generated_source = dependent_result.generated_source + "\n\n" + block_translation_result.generated_source

                command_translation_result.invocation = "#{block_translation_result.function_name}(#{Constants::ENV_IDENTIFIER}, #{dependent_result.invocation})"
                command_translation_result.size = dependent_result.size
                command_translation_result.return_type = block_translation_result.result_type
                command_translation_result.block_size = command.block_size

                command_translation_result
            end

            def visit_array_stencil_command(command)
                dependent_result = super

                command_translation_result = CommandTranslationResult.new(@environment_builder)

                # Set up parameters of block
                block_parameters = {}
                for name in command.block_parameter_names
                    block_parameters[name] = dependent_result.return_type
                end

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameter_types: block_parameter_types,
                    environment_builder: @environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                command_translation_result.generated_source = dependent_result.generated_source + "\n\n" + block_translation_result.generated_source

                command_translation_result.invocation = "#{block_translation_result.function_name}(#{Constants::ENV_IDENTIFIER}, #{dependent_result.invocation})"
                command_translation_result.size = dependent_result.size
                command_translation_result.return_type = block_translation_result.result_type
                command_translation_result.block_size = command.block_size

                command_translation_result
            end
        end

        # Retrieves all base arrays and registers them with the {EnvironmentBuilder}. Yhis functionality is in a separate class to avoid scattering with object tracer calls.
        class BaseArrayRegistrator < Symbolic::Visitor
            def initialize(environment_builder, object_tracer)
                @environment_builder = environment_builder
                @object_tracer = object_tracer
            end

            def visit_array_identity_command(command)
                need_union_type = !command.base_type.is_singleton?
                transformed_base_array = @object_tracer.convert_base_array(command.target, need_union_type)
                @environment_builder.add_base_array(command.unique_id, transformed_base_array)
            end
        end

        class << self
            def translate_command(command)
                environment_builder = EnvironmentBuilder.new

                # Run type inference for objects/classes and trace objects
                object_tracer = TypeInference::ObjectTracer.new(command)
                all_objects = object_tracer.trace_all

                # Translate command
                command_translation_result = command.accept(ArrayCommandVisitor.new(environment_builder))

                # Add SoA arrays to environment
                object_tracer.register_soa_arrays(environment_builder)

                # Add base arrays to environment
                command.accept(BaseArrayRegistrator.new(environment_builder, object_tracer))

                command_translation_result
            end
        end
    end
end
