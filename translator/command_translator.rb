require "tempfile"
require "ffi"
require_relative "translator"
require_relative "block_translator"
require_relative "../config/os_configuration"
require_relative "../symbolic/symbolic"
require_relative "../symbolic/visitor"
require_relative "../types/object_tracer"

module Ikra
    module Translator

        # Interface for transferring data to the CUDA side using FFI. Builds a struct containing all required objects (including lexical variables). Traces objects.
        class EnvironmentBuilder

            attr_accessor :objects
            attr_accessor :device_struct_allocation

            def initialize
                @objects = {}
                @device_struct_allocation = ""
            end

            # Adds an objects as a lexical variable.
            def add_object(command_id, identifier, object)
                cuda_id = "l#{command_id}_#{identifier}"
                objects[cuda_id] = object
                
                update_dev_struct_allocation(cuda_id, object)

                cuda_id
            end

            # Adds an object as a base array
            def add_base_array(command_id, object)
                cuda_id = "b#{command_id}_base"
                objects[cuda_id] = object
                cuda_id_size = "b#{command_id}_size"
                objects[cuda_id_size] = object.size

                update_dev_struct_allocation(cuda_id, object)

                cuda_id
            end

            def update_dev_struct_allocation(field, object)
                if object.class == Array
                    # Allocate new array
                    @device_struct_allocation += Translator.read_file(
                        file_name: "env_builder_copy_array.cpp",
                        replacements: { 
                            "field" => field, 
                            "host_env" => Constants::ENV_HOST_IDENTIFIER,
                            "dev_env" => Constants::ENV_DEVICE_IDENTIFIER,
                            "size_bytes" => (object.first.class.to_ikra_type.c_size * object.size).to_s})
                else
                    # Nothing to do, this case is handled by mem-copying the struct
                end
            end

            # Returns the name of the field containing the base array for a certain identity command.
            def self.base_identifier(command_id)
                "b#{command_id}_base"
            end

            def build_struct_definition
                @objects.freeze

                struct_def = "struct environment_struct\n{\n"
                @objects.each do |key, value|
                    struct_def += "    #{value.class.to_ikra_type_obj(value).to_c_type} #{key};\n"
                end
                struct_def += "};\n"

                struct_def
            end

            def build_ffi_type
                struct_layout = []
                @objects.each do |key, value|
                    struct_layout += [key.to_sym, value.class.to_ikra_type_obj(value).to_ffi_type]
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
                result.device_struct_allocation = @device_struct_allocation
                result
            end
        end

        # Result of translating a {Ikra::Symbolic::ArrayCommand}.
        class CommandTranslationResult
            attr_accessor :environment_builder
            attr_accessor :generated_source
            attr_accessor :invocation
            attr_accessor :size
            attr_accessor :return_type
            attr_accessor :kernel_classes

            def initialize
                @environment_builder = EnvironmentBuilder.new           # [EnvironmentBuilder] instance that generates the struct containing accessed lexical variables.
                @generated_source = ""                                  # [String] containing the currently generated source code.
                @invocation = "NULL"                                    # [String] source code used for invoking the block function.
                @return_type = Types::UnionType.new                     # [Types::UnionType] return type of the block.
                @size = 0                                               # [Fixnum] number of elements in base array 
                @kernel_classes = []                                    # [Array<Class>] Ruby classes that are used within the kernel

                @so_filename = ""                                       # [String] file name of shared library containing CUDA kernel
            end

            def result_size
                @size
            end

            # Compiles CUDA source code and generates a shared library.
            def compile
                # Prepare file replacements
                file_replacements = {}                                  # [Hash{String => String}] contains strings that should be replaced when reading a file 
                file_replacements["grid_dim[0]"] = "#{[size / 250, 1].max}"
                file_replacements["grid_dim[1]"] = "1"
                file_replacements["grid_dim[2]"] = "1"
                file_replacements["block_dim[0]"] = "#{size >= 250 ? 250 : size}"
                file_replacements["block_dim[1]"] = "1"
                file_replacements["block_dim[2]"] = "1"
                file_replacements["result_type"] = @return_type.singleton_type.to_c_type
                file_replacements["result_size"] = "#{result_size}"
                file_replacements["block_invocation"] = @invocation
                file_replacements["env_identifier"] = Constants::ENV_IDENTIFIER
                file_replacements["copy_env"] = @environment_builder.device_struct_allocation
                file_replacements["dev_env"] = Constants::ENV_DEVICE_IDENTIFIER
                file_replacements["host_env"] = Constants::ENV_HOST_IDENTIFIER

                # Generate source code
                source = Translator.read_file(file_name: "header.cpp", replacements: file_replacements) +
                    soa_object_array_code + 
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

                # Run compiler
                @so_filename = "#{file.path}.#{Configuration.so_suffix}"
                nvcc_command = Configuration.nvcc_invocation_string(file.path, @so_filename)

                Log.info("Compiling kernel: #{nvcc_command}")
                time_before = Time.now
                compile_status = %x(#{nvcc_command})
                Log.info("Done, took #{Time.now - time_before} s")

                if $? != 0
                    raise "nvcc failed: #{compile_status}"
                end
            end

            # Generates the CUDA code defining the arrays for the Structure-of-Arrays object layout.
            def soa_object_array_code
                definitions = @kernel_classes.map do |cls|
                    ikra_cls_type = cls.to_ikra_type
                    ikra_cls_type.accessed_inst_vars.map do |inst_var|
                        "__device__ #{ikra_cls_type.inst_vars_types[inst_var].singleton_type.to_c_type} * #{ikra_cls_type.inst_var_array_name(inst_var)};"
                    end.join("\n")
                end.join("\n")

                Translator.read_file(file_name: "soa_header.cpp", replacements: {"definitions" => definitions})
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
        end

        # A visitor traversing the tree (currently list) of symbolic array commands. Every command is converted into a {CommandTranslationResult} and possibly merged with the result of dependent (previous) results. This is how kernel fusion is implemented.
        class ArrayCommandVisitor < Symbolic::Visitor
            def visit_array_new_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    # only one block parameter (int)
                    block_parameter_types: {command.block_parameter_names.first => Types::UnionType.create_int},
                    environment_builder: command_translation_result.environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                command_translation_result.generated_source = block_translation_result.generated_source

                tid = "threadIdx.x + blockIdx.x * blockDim.x"
                command_translation_result.invocation = "#{block_translation_result.function_name}(#{Constants::ENV_IDENTIFIER}, #{tid})"
                command_translation_result.size = command.size
                command_translation_result.return_type = block_translation_result.result_type

                command_translation_result
            end

            def visit_array_identity_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                # no source code generation
                command_translation_result.invocation = "#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[threadIdx.x + blockIdx.x * blockDim.x]"
                command_translation_result.size = command.size
                command_translation_result.return_type = command.base_type
                command_translation_result.environment_builder.add_base_array(command.unique_id, command.target)

                command_translation_result
            end

            def visit_array_map_command(command)
                dependent_result = super                            # visit target (dependent) command
                command_translation_result = CommandTranslationResult.new
                command_translation_result.environment_builder = dependent_result.environment_builder.clone

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    block_parameter_types: {command.block_parameter_names.first => dependent_result.return_type},
                    environment_builder: command_translation_result.environment_builder[command.unique_id],
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                command_translation_result.generated_source = dependent_result.generated_source + "\n\n" + block_translation_result.generated_source

                command_translation_result.invocation = "#{block_translation_result.function_name}(#{Constants::ENV_IDENTIFIER}, #{dependent_result.invocation})"
                command_translation_result.size = dependent_result.size
                command_translation_result.return_type = block_translation_result.result_type

                command_translation_result
            end
        end

        class << self
            def translate_command(command)
                # Run type inference for objects/classes and trace objects
                all_objects = TypeInference::ObjectTracer.process(command)

                # Translate command
                command_translation_result = command.accept(ArrayCommandVisitor.new)
                command_translation_result.kernel_classes += all_objects.keys
                command_translation_result.kernel_classes.uniq!

                command_translation_result
            end
        end
    end
end