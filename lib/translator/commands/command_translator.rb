require_relative "../translator"
require_relative "../../config/configuration"
require_relative "../../config/os_configuration"
require_relative "../../symbolic/symbolic"
require_relative "../../symbolic/visitor"
require_relative "../../types/types"
require_relative "../input_translator"
require_relative "../../custom_weak_cache"

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

                attr_reader :command

                def initialize(execution: "", result:, command:)
                    @execution = execution
                    @command = command
                    @result = result;
                end

                def result_type
                    return command.result_type
                end
            end

            # A cache mapping commands to program builders.
            @@builder_cache = CustomWeakHash.new(comparator_selector: :has_same_code?)

            # Entry point for translator. Returns a [ProgramBuilder], which contains all
            # required information for compiling and executing the CUDA program.
            def self.translate_command(command)
                if !@@builder_cache.include?(command)
                    command_translator = self.new(root_command: command)
                    command_translator.start_translation
                    @@builder_cache[command] = command_translator.program_builder
                else
                    @@builder_cache[command].reinitialize(root_command: command)
                end

                return @@builder_cache[command]
            end

            attr_reader :environment_builder
            attr_reader :kernel_launcher_stack
            attr_reader :program_builder
            attr_reader :object_tracer
            attr_reader :root_command

            def initialize(root_command:)
                @kernel_launcher_stack = []
                @environment_builder = EnvironmentBuilder.new

                # Select correct program builder based on command type
                @program_builder = ProgramBuilder.new(
                    environment_builder: environment_builder, 
                    root_command: root_command)

                @root_command = root_command
            end

            def start_translation
                Log.info("CommandTranslator: Starting translation...")

                # Trace all objects
                @object_tracer = TypeInference::ObjectTracer.new(root_command)
                all_objects = object_tracer.trace_all


                # --- Translate ---

                # Create new kernel launcher
                push_kernel_launcher

                # Translate the command (might create additional kernels)
                result = root_command.accept(self)

                # Add kernel builder to ProgramBuilder
                pop_kernel_launcher(result)

                # --- End of Translation ---


                # Add SoA arrays to environment
                object_tracer.register_soa_arrays(environment_builder)
            end

            def kernel_launcher
                return kernel_launcher_stack.last
            end

            def kernel_builder
                return kernel_launcher_stack.last.kernel_builder
            end


            # --- Actual Visitor parts stars here ---

            def visit_array_command(command)
                if command.keep && !command.has_previous_result?
                    # Create slot for result pointer on GPU in env
                    environment_builder.allocate_previous_pointer(command.unique_id)
                end
            end

            def push_kernel_launcher(kernel_builder: nil, kernel_launcher: nil)
                if kernel_builder != nil && kernel_launcher == nil
                    @kernel_launcher_stack.push(KernelLauncher.new(kernel_builder))
                elsif kernel_builder == nil && kernel_launcher != nil
                    @kernel_launcher_stack.push(kernel_launcher)
                elsif kernel_builder == nil && kernel_launcher == nil
                    # Default: add new kernel builder
                    @kernel_launcher_stack.push(KernelLauncher.new(KernelBuilder.new))
                else
                    raise ArgumentError.new("kernel_builder and kernel_laucher given but only expected one")
                end
            end

            # Pops a KernelBuilder from the kernel builder stack. This method is called when all
            # blocks (parallel sections) for that kernel have been translated, i.e., the kernel
            # is fully built.
            def pop_kernel_launcher(command_translation_result)
                previous_launcher = kernel_launcher_stack.pop

                kernel_builder = previous_launcher.kernel_builder
                kernel_builder.block_invocation = command_translation_result.result
                kernel_builder.execution = command_translation_result.execution
                kernel_builder.result_type = command_translation_result.result_type

                if previous_launcher == nil
                    raise AssertionError.new("Attempt to pop kernel launcher, but stack is empty")
                end

                program_builder.add_kernel_launcher(previous_launcher)

                return previous_launcher
            end

            def translate_entire_input(command)
                input_translated = command.input.each_with_index.map do |input, index|
                    input.translate_input(
                        parent_command: command,
                        command_translator: self,
                        # Assuming that every input consumes exactly one parameter
                        start_eat_params_offset: index)
                end

                return EntireInputTranslationResult.new(input_translated)
            end

            # Processes a [Symbolic::Input] objects, which contains a reference to a command
            # object and information about how elements are accessed. If elements are only
            # accessed according to the current thread ID, this input can be fused. Otherwise,
            # a new kernel will be built.
            def translate_input(input)
                previous_result = ""

                if input.command.has_previous_result?
                    # Read previously computed (cached) value
                    Log.info("Reusing kept result for command #{input.command.unique_id}: #{input.command.gpu_result_pointer}")

                    environment_builder.add_previous_result(
                        input.command.unique_id, input.command.gpu_result_pointer)
                    environment_builder.add_previous_result_type(
                        input.command.unique_id, input.command.result_type)

                    cell_access = ""
                    if input.pattern == :tid
                        cell_access = "[_tid_]"
                    end

                    kernel_launcher.configure_grid(input.command.size)
                    previous_result = CommandTranslationResult.new(
                        execution: "",
                        result: "((#{input.command.result_type.to_c_type} *)(_env_->" + "prev_#{input.command.unique_id}))#{cell_access}",
                        command: input.command)

                    if input.pattern == :tid
                        return previous_result
                    else
                    end
                end

                if input.pattern == :tid
                    # Stay in current kernel                    
                    return input.command.accept(self)
                elsif input.pattern == :entire
                    if !input.command.has_previous_result?
                        # Create new kernel
                        push_kernel_launcher

                        previous_result = input.command.accept(self)
                        previous_result_kernel_var = kernel_launcher.kernel_result_var_name
                        
                        pop_kernel_launcher(previous_result)
                    else
                        kernel_launcher.use_cached_result(
                            input.command.unique_id, input.command.result_type) 
                        previous_result_kernel_var = "prev_" + input.command.unique_id.to_s
                    end

                    # Add parameter for previous input to this kernel
                    kernel_launcher.add_previous_kernel_parameter(Variable.new(
                        name: previous_result_kernel_var,
                        type: previous_result.result_type))

                    # This is a root command for this kernel, determine grid/block dimensions
                    kernel_launcher.configure_grid(input.command.size, block_size: input.command.block_size)

                    kernel_translation = CommandTranslationResult.new(
                        result: previous_result_kernel_var,
                        command: input.command)

                    return kernel_translation
                else
                    raise NotImplementedError.new("Unknown input pattern: #{input.pattern}")
                end
            end

            def build_command_translation_result(
                execution: "", result:, command:)

                result_type = command.result_type
                unique_id = command.unique_id

                if command.keep
                    # Store result in global array
                    # TODO: Remove DEBUG
                    command_result = Constants::TEMP_RESULT_IDENTIFIER + unique_id.to_s
                    command_execution = execution + "\n        " + result_type.to_c_type + " " + command_result + " = " + result + ";"

                    kernel_builder.add_cached_result(unique_id.to_s, result_type)
                    kernel_launcher.add_cached_result(unique_id.to_s, result_type)
                    environment_builder.add_previous_result_type(unique_id, result_type)
                else
                    command_result = result
                    command_execution = execution
                end

                command_translation = CommandTranslationResult.new(
                    execution: command_execution,
                    result: command_result,
                    command: command)
            end
        end
    end
end

require_relative "array_combine_command"
require_relative "array_index_command"
require_relative "array_identity_command"
require_relative "array_reduce_command"
require_relative "array_stencil_command"
require_relative "array_zip_command"
require_relative "../host_section/array_host_section_command"

require_relative "../program_builder"
require_relative "../kernel_launcher/kernel_launcher"
