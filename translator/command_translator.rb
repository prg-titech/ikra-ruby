require "ffi"
require_relative "translator"
require_relative "../config/os_configuration"
require "logger"
require "tempfile"

module Ikra
    Log = Logger.new(STDOUT)

    module Translator

        # Interface for transferring data to the CUDA side using FFI. Builds a struct containing all required objects (including lexical variables). Traces objects.
        class EnvironmentBuilder

            attr_accessor :objects

            def initialize
                @objects = {}
            end

            # Adds an objects as a lexical variable.
            def add_object(command_id, identifier, object)
                cuda_id = "l#{command_id}_#{identifier}"
                objects[cuda_id] = object
                cuda_id
            end

            # Adds an object as a base array
            def add_base_array(command_id, object)
                cuda_id = "b#{command_id}_base"
                objects[cuda_id] = object
                cuda_id
            end

            def build_struct_definition
                @objects.freeze

                struct_def = "struct environment_struct\n"
                @objects.each do |key, value|
                    struct_def += "    #{value.to_ikra_type.to_c_type} #{key};\n"
                end
                struct_def += "};\n"

                struct_def
            end

            def build_ffi_type
                struct_layout = []
                @objects.each do |key, value|
                    struct_layout += [key.to_sym, value.class.to_ikra_type.to_ffi_type]
                end

                struct_type = Class.new(FFI::Struct)
                struct_type.layout(*struct_layout)

                struct_type
            end

            def build_ffi_object
                struct_type = build_ffi_type
                struct = struct_type.new

                @objects.each do |key, value|
                    # TODO: need proper Array handling
                    if value.class == Array
                        # Check first element to determine type of array
                        # TODO: check for polymorphic
                        inner_type = value.first.class.to_ikra_type
                        array_ptr = FFI::MemoryPointer.new(value.size * inner_type.c_size)

                        if inner_type == Types::PrimitiveType::Int
                            array_ptr.put_array_of_int(0, value)
                        elsif inner_type == Types::PrimitiveType::Float
                            array_ptr.put_array_of_float(0, value)
                        else
                            raise NotImplementedError
                        end

                        struct[key.to_sym] = array_ptr
                    else
                        struct[key.to_sym] = value
                    end
                end

                struct.to_ptr
            end

            def [](command_id)
                CurriedBuilder.new(self, command_id)
            end

            class CurriedBuilder
                def initialize(builder, command_id)
                    @builder = builder
                    @command_id = command_id
                end

                def add_object(identifier, object)
                    @builder.add_object(@command_id, identifier, object)
                end

                def add_base_array(object)
                    @builder.add_base_array(@command_id, object)
                end
            end

            def clone
                result = self.class.new
                result.objects = @objects.clone
                result
            end
        end

        # Result of translating a [Symbolic::ArrayCommand].
        class CommandTranslationResult
            attr_accessor :environment_builder
            attr_accessor :generated_source
            attr_accessor :invocation
            attr_accessor :size
            attr_accessor :return_type
            attr_accessor :base_arrays

            def initialize
                @environment_builder = EnvironmentBuilder.new           # [EnvironmentBuilder] instance that generates the struct containing accessed lexical variables.
                @generated_source = ""                                  # [String] containing the currently generated source code.
                @invocation = "NULL"                                    # [String] source code used for invoking the block function.
                @return_type = Types::UnionType.new                     # [Types::UnionType] return type of the block.
                @size = 0                                               # [Fixnum] number of elements in base array
                @base_arrays = {}                                       # [Hash{Symbol => Array}] hash mapping identifier in CUDA code to array

                @file_replacements = {}                                 # [Hash{String => String}] contains strings that should be replaced when reading a file 
                @so_filename = ""                                       # [String] file name of shared library containing CUDA kernel
            end

            def result_size
                # TODO: this is a dirty hack
                @base_arrays.values.first.size
            end

            def compile
                # Prepare file replacements
                @file_replacements["grid_dim[0]"] = "#{[size / 250, 1].max}"
                @file_replacements["grid_dim[1]"] = "1"
                @file_replacements["grid_dim[2]"] = "1"
                @file_replacements["block_dim[0]"] = "#{size >= 250 ? 250 : size}"
                @file_replacements["block_dim[1]"] = "1"
                @file_replacements["block_dim[2]"] = "1"
                @file_replacements["result_type"] = @return_type.singleton_type.to_c_type
                @file_replacements["result_size"] = "#{result_size}"

                # Generate source code
                source = read_file("header.cpp")
                    + @envionment_builder.struct_definition("environment_struct")
                    + read_file("kernel_launcher.cpp")
                    + @generated_source

                # Write source code to temporary file
                file = Tempfile.new(["ikra_kernel", ".cu"])
                file.write(source)
                file.close

                # Run compiler
                so_filename = "#{file.path}.#{Configuation.so_suffix}"])
                nvcc_command = Configuration.nvcc_invocation_string(file.path, so_filename)

                Log.info("Compiling kernel: #{nvcc_command}")
                time_before = Time.now
                compile_status = %x(#{nvcc_command})
                Log.info("Done, took #{Time.now - time_before} s")

                if $? != 0
                    raise "nvcc failed: #{compile_status}"
                end
            end

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
                result = ffi_interface.launch_kernel(environment_object)
                Log.info("Kernel time: #{Time.now - time_before} s")

                if return_type.singleton_type == Types::PrimitiveType::Int
                    result.read_array_of_int(result_size)
                elsif return_type.singleton_type == Types::PrimitiveType::Float
                    result.read_array_of_float(result_size)
                else
                    raise NotImplementedError
                end
            end

            private

            def read_file(file_name)
                full_name = File.expand_path("resources/cuda/#{file_name}", File.dirname(File.dirname(File.expand_path(__FILE__))))
                contents = File.open(full_name, "rb").read

                @file_replacements.each do |s1, s2|
                    contents = contents.gsub("/*#{s1}*/", s2)
                end

                contents
            end
        end

        class ArrayCommandVisitor < Symbolic::Visitor
            def visit_array_new_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    # only one block parameter (int)
                    block_parameter_types: {command.block_parameter_names.first => Types::UnionType.create_int},
                    environment_builder: command_translation_result.environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals)

                command_translation_result.generated_source = block_translation_result.generated_source

                tid = "threadIdx.x + blockIdx.x * blockDim.x"
                command_translation_result.invocation = "#{block_translation_result.function_name}(#{EnvParameterName}, #{tid})"
                command_translation_result.size = command.size
                command_translation_result.return_type = block_translation_result.result_type

                command_translation_result
            end

            def visit_array_identity_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                # no source code generation
                command_translation_result.invocation = "_input_k#{command.unique_id}_[threadIdx.x + blockIdx.x * blockDim.x]"
                command_translation_result.size = command.size
                command_translation_result.return_type = command.base_type
                command_translation_result.base_arrays["_input_k#{command.unique_id}_".to_sym] = command.target
                command_translation_result.environment_builder.add_base_array(command.unique_id, command.target)

                command_translation_result
            end

            def visit_array_map_command(command)
                dependent_result = super                            # visit target (dependent) command
                command_translation_result = CommandTranslationResult.new
                command_translation_result.environment_builder = dependent_result.environment_builder.clone
                command_translation_result.base_arrays = dependent_result.base_arrays.clone

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    block_parameter_types: {command.block_parameter_names.first => dependent_result.return_type},
                    environment_builder: command_translation_result.environment_builder[command.unique_id],
                    lexical_variables: command.lexical_variables)

                command_translation_result.generated_source = dependent_result.generated_source + "\n\n" + block_translation_result.generated_source

                command_translation_result.invocation = "#{block_translation_result.function_name}(#{EnvParameterName}, #{dependent_result.invocation})"
                command_translation_result.size = dependent_result.size
                command_translation_result.return_type = block_translation_result.result_type

                command_translation_result
            end
        end

        class << self
            def translate_command(command)
                command.accept(ArrayCommandVisitor.new)
            end
        end
    end
end