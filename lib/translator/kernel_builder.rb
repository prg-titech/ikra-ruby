module Ikra
    module Translator
        class CommandTranslator

            # Builds a CUDA kernel. This class is responsible for generating the kernel function
            # itself (not the block functions/methods though) and the code that invokes the kernel.
            #
            # For example:
            # __global__ void kernel(env_t *_env_, int *_result_, int *_previous_1_*, ...) { ... }
            #
            # And the launcher:
            # kernel<<<..., ...>>>(env, result, d_a, ...);
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

                # Number of threads (elements to be processed)
                attr_accessor :num_threads

                # Block/grid dimensions (should be 1D)
                attr_accessor :grid_dim
                attr_accessor :block_dim

                # These fields can be read after building the kernel
                attr_reader :host_result_var_name

                attr_reader :kernel_result_var_name

                def initialize
                    @methods = []
                    @blocks = []
                    @previous_kernel_input = []
                    @block_invocation = nil
                    @num_threads = nil
                    @write_back_to_host = false
                    @kernel_name = "kernel_" + CommandTranslator.next_unique_id.to_s
                    @kernel_result_var_name = "_kernel_result_" + CommandTranslator.next_unique_id.to_s
                end

                # Configures grid size and block size. Also sets number of threads.
                def configure_grid(size)
                    @grid_dim = [size.fdiv(250).ceil, 1].max.to_s
                    @block_dim = (size >= 250 ? 250 : size).to_s
                    @num_threads = size
                end

                def write_back_to_host!
                    @write_back_to_host = true
                    @host_result_var_name = @kernel_result_var_name + "_host"
                end

                def write_back_to_host?
                    return @write_back_to_host
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

                def assert_ready_to_build
                    required_values = [:block_invocation, :result_type, :num_threads, :grid_dim, :block_dim]

                    for selector in required_values
                        if send(selector) == nil
                            raise "Not ready to build (KernelBuilder): #{selector} is not set"
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
                    p_result = result_type.to_c_type + " *" + Constants::RESULT_IDENTIFIER

                    previous_kernel_params = []
                    for var in previous_kernel_input
                        previous_kernel_params.push(var.type.to_c_type + " *" + var.name.to_s)
                    end

                    parameters = ([p_env, p_result] + previous_kernel_params).join(", ")

                    # Build kernel
                    return Translator.read_file(file_name: "kernel.cpp", replacements: {
                        "block_invocation" => block_invocation,
                        "execution" => execution,
                        "kernel_name" => kernel_name,
                        "num_threads" => num_threads.to_s,
                        "parameters" => parameters})
                end

                # Build the code that launches this kernel. The generated code performs the
                # following steps:
                #
                # 1. Allocate device memory for the result.
                # 2. If result should be written back: Allocate host memory for the result.
                # 3. Launch the kernel (+ error checking, synchronization)
                # 4. If result should be written back: Copy result back to host memory.
                def build_kernel_lauchner
                    Log.info("Building kernel launcher (write_back=#{write_back_to_host?})")

                    assert_ready_to_build

                    result = ""

                    # Allocate device memory for kernel result
                    result = result + Translator.read_file(file_name: "allocate_device_memory.cpp", replacements: {
                        "name" => kernel_result_var_name,
                        "bytes" => "(#{result_type.c_size} * #{num_threads})",
                        "type" => result_type.to_c_type})


                    if write_back_to_host?
                        # Allocate host memory for kernel result
                        result = result + Translator.read_file(file_name: "allocate_host_memory.cpp", replacements: {
                            "name" => @host_result_var_name,
                            "bytes" => "(#{result_type.c_size} * #{num_threads})",
                            "type" => result_type.to_c_type})
                    end

                    # Build arguments
                    a_env = Constants::ENV_DEVICE_IDENTIFIER
                    a_result = kernel_result_var_name

                    previous_kernel_args = []
                    for var in previous_kernel_input
                        previous_kernel_args.push(var.name.to_s)
                    end

                    arguments = ([a_env, a_result] + previous_kernel_args).join(", ")

                    # Launch kernel
                    result = result + Translator.read_file(file_name: "launch_kernel.cpp", replacements: {
                        "kernel_name" => kernel_name,
                        "arguments" => arguments,
                        "grid_dim" => grid_dim,
                        "block_dim" => block_dim})

                    if write_back_to_host?
                        # Memcpy kernel result from device to host
                        result = result + Translator.read_file(file_name: "memcpy_device_to_host.cpp", replacements: {
                            "host_name" => @host_result_var_name,
                            "device_name" => @kernel_result_var_name,
                            "bytes" => "(#{result_type.c_size} * #{num_threads})"})
                    end

                    return result
                end
            end
        end
    end
end
