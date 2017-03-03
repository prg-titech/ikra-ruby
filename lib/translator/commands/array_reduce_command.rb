module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_reduce_command(command)
                Log.info("Translating ArrayReduceCommand [#{command.unique_id}]")

                super

                if command.input.size != 1
                    raise AssertionError.new("Expected exactly one input for ArrayReduceCommand")
                end

                # Process dependent computation (receiver)
                input = translate_entire_input(command)

                block_size = command.block_size

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id,
                    entire_input_translation: input)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Add "odd" parameter to the kernel which is needed for reduction
                kernel_builder.add_additional_parameters(Constants::ODD_TYPE + " " + Constants::ODD_IDENTIFIER)

                # Number of elements that will be reduced
                num_threads = command.input_size

                if num_threads.is_a?(Fixnum)
                    # Easy case: Number of required reductions known statically

                    odd = (num_threads % 2 == 1).to_s

                    # Number of threads needed for reduction
                    num_threads = num_threads.fdiv(2).ceil

                    previous_result_kernel_var = input.result.first
                    first_launch = true
                    
                    # While more kernel launches than one are needed to finish reduction
                    while num_threads >= block_size + 1
                        # Launch new kernel (with same kernel builder)
                        push_kernel_launcher(kernel_builder: kernel_builder)
                        # Configure kernel with correct arguments and grid
                        kernel_launcher.add_additional_arguments(odd)
                        kernel_launcher.configure_grid(num_threads, block_size: block_size)
                        
                        # First launch of kernel is supposed to allocate new memory, so only reuse memory after first launch 
                        if first_launch
                            first_launch = false
                        else
                            kernel_launcher.reuse_memory!(previous_result_kernel_var)
                        end

                        previous_result_kernel_var = kernel_launcher.kernel_result_var_name

                        pop_kernel_launcher(input.command_translation_result(0))

                        # Update number of threads needed
                        num_threads = num_threads.fdiv(block_size).ceil
                        odd = (num_threads % 2 == 1).to_s
                        num_threads = num_threads.fdiv(2).ceil
                    end

                    # Configuration for last launch of kernel
                    kernel_launcher.add_additional_arguments(odd)
                    kernel_launcher.configure_grid(num_threads, block_size: block_size)
                else
                    # More difficult case: Have to generate loop for reductions

                    # Add one regular kernel launcher for setting up the memory etc.
                    odd_first = "(#{num_threads} % 2 == 1)"
                    num_threads_first = "((int) ceil(#{num_threads} / 2.0))"
                    push_kernel_launcher(kernel_builder: kernel_builder)
                    kernel_launcher.add_additional_arguments(odd_first)
                    kernel_launcher.configure_grid(num_threads_first, block_size: block_size)
                    previous_result_kernel_var = kernel_launcher.kernel_result_var_name
                    pop_kernel_launcher(input.command_translation_result(0))

                    # Add loop
                    # Set up state (variables that are updated inside the loop)
                    # 1. Calculate number of elements from previous computation
                    # 2. Check if odd number
                    # 3. Calculate number of threads that we need
                    loop_setup = "int _num_elements = ceil(#{num_threads_first} / (double) #{block_size});\nbool _next_odd = _num_elements % 2 == 1;\nint _next_threads = ceil(_num_elements / 2.0);\n"

                    # Update loop state after iteration
                    update_loop = "_num_elements = ceil(_next_threads / (double) #{block_size});\nbool _next_odd = _num_elements % 2 == 0;\n_next_threads = ceil(_num_elements / 2.0);\n"

                    push_kernel_launcher(kernel_launcher: WhileLoopKernelLauncher.new(
                        kernel_builder: kernel_builder,
                        condition: "_num_elements > 1",
                        before_loop: loop_setup,
                        post_iteration: update_loop))

                    kernel_launcher.add_additional_arguments("_next_odd")
                    kernel_launcher.configure_grid("_next_threads", block_size: block_size)
                    #pop_kernel_launcher(input.command_translation_result(0))
                end

                if !first_launch
                    kernel_launcher.reuse_memory!(previous_result_kernel_var)
                end

                command_execution = Translator.read_file(file_name: "reduce_body.cpp", replacements: {
                    "previous_result" => input.result.first,
                    "block_name" => block_translation_result.function_name,
                    "arguments" => Constants::ENV_IDENTIFIER,
                    "block_size" => block_size.to_s,
                    "temp_result" => Constants::TEMP_RESULT_IDENTIFIER,
                    "odd" => Constants::ODD_IDENTIFIER,
                    "type" => command.result_type.to_c_type,
                    "num_threads" => Constants::NUM_THREADS_IDENTIFIER})

                command_translation = CommandTranslationResult.new(
                    execution: command_execution,
                    result:  Constants::TEMP_RESULT_IDENTIFIER,
                    command: command)

                Log.info("DONE translating ArrayReduceCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
