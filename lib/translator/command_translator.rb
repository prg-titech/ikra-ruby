require_relative "translator"
require_relative "../config/configuration"
require_relative "../config/os_configuration"
require_relative "../symbolic/symbolic"
require_relative "../symbolic/visitor"
require_relative "../types/types"

module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            @@unique_id = 0

            def self.next_unique_id
                @@unique_id + @@unique_id + 1
                return @@unique_id
            end

            class CommandTranslationResult
                attr_reader :command_invocation
                attr_reader :return_type

                def initialize(command_invocation:, return_type:)
                    @command_invocation = command_invocation
                    @return_type = return_type
                end
            end

            # Entry point for translator. Returns a [ProgramBuilder], which contains all
            # required information for compiling and executing the CUDA program.
            def self.translate_command(command)
                command_translator = self.new
                command_translator.translate_command(command)
                return command_translator.program_builder
            end

            attr_reader :environment_builder
            attr_reader :kernel_builder_stack
            attr_reader :program_builder
            attr_reader :object_tracer

            def initialize
                @kernel_builder_stack = []
                @environment_builder = EnvironmentBuilder.new
                @program_builder = ProgramBuilder.new(environment_builder: environment_builder)
            end

            def translate_command(command)
                Log.info("CommandTranslator: Starting translation...")

                # Trace all objects
                @object_tracer = TypeInference::ObjectTracer.new(command)
                all_objects = object_tracer.trace_all


                # --- Translate ---

                # Create new kernel builder
                push_kernel_builder

                # Result of this kernel should be written back to the host
                kernel_builder.write_back_to_host!

                # Translate the command (might create additional kernels)
                result = command.accept(self)

                # Add kernel builder to ProgramBuilder
                pop_kernel_builder(result)

                # --- End of Translation ---


                # Add SoA arrays to environment
                object_tracer.register_soa_arrays(environment_builder)
            end

            def kernel_builder
                return kernel_builder_stack.last
            end


            # --- Actual Visitor parts stars here ---

            # Translate the block of an `Array.pnew` section.
            def visit_array_new_command(command)
                Log.info("Translating ArrayNewCommand [#{command.unique_id}]")

                # This is a root command, determine grid/block dimensions
                grid_dim = [command.size.fdiv(250).ceil, 1].max
                block_dim = command.size >= 250 ? 250 : command.size
                kernel_builder.grid_dim = grid_dim.to_s
                kernel_builder.block_dim = block_dim.to_s
                kernel_builder.num_threads = command.size

                # Thread ID is always int
                parameter_types = {command.block_parameter_names.first => Types::UnionType.create_int}

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameter_types: parameter_types,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                command_translation = CommandTranslationResult.new(
                    command_invocation: block_translation_result.function_name + "(_env_, _tid_)",
                    return_type: block_translation_result.result_type)
                
                Log.info("DONE translating ArrayNewCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_map_command(command)
                Log.info("Translating ArrayMapCommand [#{command.unique_id}]")

                # Process dependent computation (receiver), returns [CommandTranslationResult]
                input_translated = translate_input(command.input.first)

                # Take return type from previous computation
                parameter_types = {command.block_parameter_names.first => input_translated.return_type}

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameter_types: parameter_types,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                command_translation = CommandTranslationResult.new(
                    command_invocation: block_translation_result.function_name + "(_env_, #{input_translated.command_invocation})",
                    return_type: block_translation_result.result_type)


                Log.info("DONE translating ArrayCombineCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_combine_command(command)
                Log.info("Translating ArrayCombineCommand [#{command.unique_id}]")

                # Process dependent computation (receiver), returns [CommandTranslationResult]
                input_translated = command.input.map do |input|
                    translate_input(input)
                end
                # Map translated input on return types to prepare hashing of parameter_types
                return_types = input_translated.map do |input|
                    input.return_type
                end

                # Take return types from previous computation
                parameter_types = Hash[command.block_parameter_names.zip(return_types)]

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameter_types: parameter_types,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Build command invocation string
                command_invocation = block_translation_result.function_name + "(_env_"

                input_translated.each do |input| 
                    command_invocation <<  ", #{input.command_invocation}"
                end

                command_invocation << ")"

                command_translation = CommandTranslationResult.new(
                    command_invocation: command_invocation,
                    return_type: block_translation_result.result_type)


                Log.info("DONE translating ArrayMapCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_identity_command(command)
                Log.info("Translating ArrayIdentityCommand [#{command.unique_id}]")

                # This is a root command, determine grid/block dimensions
                grid_dim = [command.size.fdiv(250).ceil, 1].max
                block_dim = command.size >= 250 ? 250 : command.size
                kernel_builder.grid_dim = grid_dim.to_s
                kernel_builder.block_dim = block_dim.to_s
                kernel_builder.num_threads = command.size

                # Add base array to environment
                need_union_type = !command.base_type.is_singleton?
                transformed_base_array = object_tracer.convert_base_array(
                    command.input.first.command, need_union_type)
                environment_builder.add_base_array(command.unique_id, transformed_base_array)

                command_translation = CommandTranslationResult.new(
                    command_invocation:"#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[_tid_]",
                    return_type: command.base_type)

                Log.info("DONE translating ArrayIdentityCommand [#{command.unique_id}]")

                return command_translation
            end

            def push_kernel_builder
                @kernel_builder_stack.push(KernelBuilder.new)
            end

            # Pops a KernelBuilder from the kernel builder stack. This method is called when all
            # blocks (parallel sections) for that kernel have been translated, i.e., the kernel
            # is fully built.
            def pop_kernel_builder(command_translation_result)
                previous_builder = kernel_builder_stack.pop
                previous_builder.block_invocation = command_translation_result.command_invocation
                previous_builder.result_type = command_translation_result.return_type

                if previous_builder == nil
                    raise "Attempt to pop kernel builder, but stack is empty"
                end

                program_builder.add_kernel(previous_builder)
            end

            # Processes a [Symbolic::Input] objects, which contains a reference to a command
            # object and information about how elements are accessed. If elements are only
            # accessed according to the current thread ID, this input can be fused. Otherwise,
            # a new kernel will be built.
            def translate_input(input)
                if input.pattern == :tid
                    # Stay in current kernel
                    return input.command.accept(self)
                elsif input.pattern == :entire
                    # Create new kernel
                    push_kernel_builder
                    result = input.command.accept(self)
                    # TODO: Should return array for old kernel here
                    pop_kernel_builder(result)
                    return result
                else
                    raise "Unknown input pattern: #{input.pattern}"
                end
            end
        end
    end
end

require_relative "program_builder"
require_relative "kernel_builder"