require_relative "translator"
require_relative "../config/configuration"
require_relative "../config/os_configuration"
require_relative "../symbolic/symbolic"
require_relative "../symbolic/visitor"
require_relative "../types/types"
require_relative "input_translator"

module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            @@unique_id = 0

            def self.next_unique_id
                @@unique_id = @@unique_id + 1
                return @@unique_id
            end

            class CommandTranslationResult
                # Source code that performs the computation of this command for one thread. May 
                # consist of multiple statement. Optional.
                attr_reader :execution

                # Source code that returns the result of the computation. If the computation can
                # be expressed in a single expression, this string can contain the entire
                # computation and `execution` should then be empty.
                attr_reader :result

                attr_reader :return_type

                def initialize(execution: "", result:, return_type:)
                    @execution = execution
                    @return_type = return_type
                    @result = result;
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
                kernel_builder.configure_grid(command.size)

                # Thread ID is always int
                parameters = [Variable.new(
                    name: command.block_parameter_names.first,
                    type: Types::UnionType.create_int)]

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                command_translation = CommandTranslationResult.new(
                    result: block_translation_result.function_name + "(_env_, _tid_)",
                    return_type: block_translation_result.result_type)
                
                Log.info("DONE translating ArrayNewCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_combine_command(command)
                Log.info("Translating ArrayCombineCommand [#{command.unique_id}]")

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input_translated = command.input.each_with_index.map do |input, index|
                    input.translate_input(
                        command: command,
                        command_translator: self,
                        start_eat_params_offset: index)
                end

                # Get all parameters
                block_parameters = input_translated.map do |input|
                    input.parameters
                end.reduce(:+)

                # Ger all pre-execution statements
                pre_execution = input_translated.map do |input|
                    input.pre_execution
                end.reduce(:+)

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: block_parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    pre_execution: pre_execution,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Build command invocation string
                command_args = (["_env_"] + input_translated.map do |input|
                    input.command_translation_result.result
                end).join(", ")

                command_result = block_translation_result.function_name + "(" + command_args + ")"

                input_execution = input_translated.map do |input|
                    input.command_translation_result.execution
                end.join("\n\n")

                command_translation = CommandTranslationResult.new(
                    execution: input_execution,
                    result: command_result,
                    return_type: block_translation_result.result_type)


                Log.info("DONE translating ArrayCombineCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_identity_command(command)
                Log.info("Translating ArrayIdentityCommand [#{command.unique_id}]")

                # This is a root command, determine grid/block dimensions
                kernel_builder.configure_grid(command.size)

                # Add base array to environment
                need_union_type = !command.base_type.is_singleton?
                transformed_base_array = object_tracer.convert_base_array(
                    command.input.first.command, need_union_type)
                environment_builder.add_base_array(command.unique_id, transformed_base_array)

                command_translation = CommandTranslationResult.new(
                    result: "#{Constants::ENV_IDENTIFIER}->#{EnvironmentBuilder.base_identifier(command.unique_id)}[_tid_]",
                    return_type: command.base_type)

                Log.info("DONE translating ArrayIdentityCommand [#{command.unique_id}]")

                return command_translation
            end

            def visit_array_stencil_command(command)
                Log.info("Translating ArrayStencilCommand [#{command.unique_id}]")

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input_translated = command.input.first.translate_input(
                    command: command,
                    command_translator: self)

                # Count number of parameters
                num_parameters = command.offsets.size

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: input_translated.parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id,
                    pre_execution: input_translated.pre_execution,
                    override_block_parameters: input_translated.override_block_parameters)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # `previous_result` should be an expression returning the array containing the
                # result of the previous computation.
                previous_result = input_translated.command_translation_result.result

                arguments = ["_env_"]

                # Pass values from previous computation that are required by this thread.
                for i in 0...num_parameters
                    arguments.push("#{previous_result}[_tid_ + #{command.offsets[i]}]")
                end

                argument_str = arguments.join(", ")
                stencil_computation = block_translation_result.function_name + "(#{argument_str})"

                temp_var_name = "temp_stencil_#{CommandTranslator.next_unique_id}"

                # The following template checks if there is at least one index out of bounds. If
                # so, the fallback value is used. Otherwise, the block is executed.
                command_execution = Translator.read_file(file_name: "stencil_body.cpp", replacements: {
                    "execution" => input_translated.command_translation_result.execution,
                    "temp_var" => temp_var_name,
                    "result_type" => block_translation_result.result_type.to_c_type,
                    "min_offset" => command.min_offset.to_s,
                    "max_offset" => command.max_offset.to_s,
                    "thread_id" => "_tid_",
                    "input_size" => command.input.first.command.size.to_s,
                    "out_of_bounds_fallback" => command.out_of_range_value.to_s,
                    "stencil_computation" => stencil_computation})

                command_translation = CommandTranslationResult.new(
                    execution: command_execution,
                    result: temp_var_name,
                    return_type: block_translation_result.result_type)

                Log.info("DONE translating ArrayStencilCommand [#{command.unique_id}]")

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
                previous_builder.block_invocation = command_translation_result.result
                previous_builder.execution = command_translation_result.execution
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

                    previous_result = input.command.accept(self)
                    previous_result_kernel_var = kernel_builder.kernel_result_var_name
                    
                    pop_kernel_builder(previous_result)

                    # Add parameter for previous input to this kernel
                    kernel_builder.add_previous_kernel_parameter(Variable.new(
                        name: previous_result_kernel_var,
                        type: previous_result.return_type))

                    # This is a root command for this kernel, determine grid/block dimensions
                    kernel_builder.configure_grid(input.command.size)

                    kernel_translation = CommandTranslationResult.new(
                        result: previous_result_kernel_var,
                        return_type: previous_result.return_type)

                    return kernel_translation
                else
                    raise "Unknown input pattern: #{input.pattern}"
                end
            end
        end
    end
end

require_relative "program_builder"
require_relative "kernel_builder"