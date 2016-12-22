module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_reduce_command(command)
                Log.info("Translating ArrayReduceCommand [#{command.unique_id}]")

                super

                # Process dependent computation (receiver)
                input_translated = command.input.first.translate_input(
                    command: command,
                    command_translator: self)

                block_size = command.block_size

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    block_parameters: input_translated.parameters,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    pre_execution: input_translated.pre_execution,
                    command_id: command.unique_id)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Add "odd" parameter to the kernel which is needed for reduction
                kernel_builder.add_additional_parameters(Constants::ODD_TYPE + " " + Constants::ODD_IDENTIFIER)

                # Number of elements that will be reduced
                num_threads = command.size
                odd = num_threads % 2 == 1
                # Number of threads needed for reduction
                num_threads = num_threads.fdiv(2).ceil

                previous_result_kernel_var = input_translated.command_translation_result.result
                first_launch = true
                
                # While more kernel launches than one are needed to finish reduction
                while num_threads >= block_size + 1
                    # Launch new kernel (with same kernel builder)
                    push_kernel_launcher(kernel_builder)
                    # Configure kernel with correct arguments and grid
                    kernel_launcher.add_additional_arguments(odd)
                    kernel_launcher.configure_grid(num_threads)
                    
                    # First launch of kernel is supposed to allocate new memory, so only reuse memory after first launch 
                    if first_launch
                        first_launch = false
                    else
                        kernel_launcher.reuse_memory!(previous_result_kernel_var)
                    end

                    previous_result_kernel_var = kernel_launcher.kernel_result_var_name

                    pop_kernel_launcher(input_translated.command_translation_result)

                    # Update number of threads needed
                    num_threads = num_threads.fdiv(block_size).ceil
                    odd = num_threads % 2 == 1
                    num_threads = num_threads.fdiv(2).ceil
                end

                # Configuration for last launch of kernel
                kernel_launcher.add_additional_arguments(odd)
                kernel_launcher.configure_grid(num_threads)

                if !first_launch
                    kernel_launcher.reuse_memory!(previous_result_kernel_var)
                end

                command_execution = Translator.read_file(file_name: "reduce_body.cpp", replacements: {
                    "previous_result" => input_translated.command_translation_result.result,
                    "block_name" => block_translation_result.function_name,
                    "arguments" => Constants::ENV_IDENTIFIER,
                    "block_size" => block_size.to_s,
                    "temp_result" => Constants::TEMP_RESULT_IDENTIFIER,
                    "odd" => Constants::ODD_IDENTIFIER,
                    "type" => block_translation_result.result_type.to_c_type,
                    "num_threads" => Constants::NUM_THREADS_IDENTIFIER})

                command_translation = CommandTranslationResult.new(
                    execution: command_execution,
                    result:  Constants::TEMP_RESULT_IDENTIFIER,
                    return_type: block_translation_result.result_type)

                Log.info("DONE translating ArrayReduceCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
