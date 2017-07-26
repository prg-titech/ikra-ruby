module Ikra
    module Translator
        class CommandTranslator

            # Builds a CUDA kernel. This class is responsible for generating the kernel function
            # itself (not the block functions/methods though).
            #
            # For example:
            # __global__ void kernel(env_t *_env_, int *_result_, int *_previous_1_*, ...) { ... }
            
            class KernelBuilder
                attr_accessor :kernel_name

                # --- Optional fields ---

                # An array of all methods that should be translated
                attr_accessor :methods

                # An array of all blocks that should be translated
                attr_accessor :blocks

                # Additional parameters that this kernel should accept (to access the result
                # of previous kernels)
                attr_accessor :previous_kernel_input

                # --- Required fields ---

                # A string returning the result of this kernel for one thread
                attr_accessor :block_invocation

                # A string containing the statements that execute the body of the kernel
                attr_accessor :execution

                # The result type of this kernel
                attr_accessor :result_type

                # Additional Parameters for certain commands that are attached to the kernel
                attr_accessor :additional_parameters

                # IDs of commands that whose results are kept on the GPU
                attr_accessor :cached_results

                def initialize
                    @methods = []
                    @blocks = []
                    @previous_kernel_input = []
                    @block_invocation = nil
                    @num_threads = nil
                    @additional_parameters = []
                    @kernel_name = "kernel_" + CommandTranslator.next_unique_id.to_s
                    @cached_results = {}
                    @execution = ""
                end
                
                # --- Prepare kernel ---

                # Adds one or multiple methods (source code strings) to this builder.
                def add_methods(*method)
                    @methods.push(*method)
                end

                # Adds a block (source code string) to this builder.
                def add_block(block)
                    @blocks.push(block)
                end

                def add_previous_kernel_parameter(parameter)
                    @previous_kernel_input.push(parameter)
                end

                # Add additional parameters to the kernel function that might be needed for some computations
                def add_additional_parameters(parameter)
                    @additional_parameters.push(parameter)
                end

                # Adds a result that has to be kept on GPU. Therefore additional memory allocations will be made
                def add_cached_result(result_id, type)
                    @cached_results[result_id] = type
                end

                def assert_ready_to_build
                    required_values = [:block_invocation, :result_type]

                    for selector in required_values
                        if send(selector) == nil
                            raise AssertionError.new(
                                "Not ready to build (KernelBuilder): #{selector} is not set")
                        end
                    end
                end


                # --- Constructor source code ---

                def build_methods
                    return @methods.join("\n\n")
                end

                def build_blocks
                    return @blocks.join("\n\n")
                end

                def build_kernel
                    Log.info("Building kernel (num_blocks=#{@blocks.size})")
                    assert_ready_to_build

                    # Build parameters
                    p_env = Constants::ENV_TYPE + " *" + Constants::ENV_IDENTIFIER
                    p_num_threads = Constants::NUM_THREADS_TYPE + " " + Constants::NUM_THREADS_IDENTIFIER
                    p_result = result_type.to_c_type + " *" + Constants::RESULT_IDENTIFIER
                    p_cached_results = cached_results.map do |result_id, type|
                        type.to_c_type + " *" + Constants::RESULT_IDENTIFIER + result_id
                    end

                    cached_results.each do |result_id, type|
                        @execution = execution + "\n" + "        " + Constants::RESULT_IDENTIFIER + result_id + "[_tid_] = " + Constants::TEMP_RESULT_IDENTIFIER + result_id + ";"
                    end

                    previous_kernel_params = []
                    for var in previous_kernel_input
                        previous_kernel_params.push(var.type.to_c_type + " *" + var.name.to_s)
                    end

                    parameters = ([p_env, p_num_threads, p_result] + p_cached_results + previous_kernel_params + additional_parameters).join(", ")

                    # Build kernel
                    return Translator.read_file(file_name: "kernel.cpp", replacements: {
                        "block_invocation" => block_invocation,
                        "execution" => execution,
                        "kernel_name" => kernel_name,
                        "parameters" => parameters,
                        "num_threads" => Constants::NUM_THREADS_IDENTIFIER,
                        "curand_seed" => Constants::CURAND_SEED})
                end
            end
        end
    end
end
