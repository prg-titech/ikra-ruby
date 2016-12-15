module Ikra
    module Translator
        class CommandTranslator

            # Builds the launch of the kernel. This class is responsible for generating the invocation of the kernel.
            #
            # For example:
            # kernel<<<..., ...>>>(env, result, d_a, ...);
            class KernelLauncher
                attr_accessor :kernel_builder

                # Additional parameters that this kernel should accept (to access the result
                # of previous kernels)
                attr_accessor :previous_kernel_input

                # Additional parameters that this kernel should accept (to access the result
                # of previous kernels)
                attr_accessor :additional_arguments

                # Number of threads (elements to be processed)
                attr_accessor :num_threads

                # Block/grid dimensions (should be 1D)
                attr_accessor :grid_dim
                attr_accessor :block_dim

                # Whether the result of this launch is written back to host
                attr_accessor :write_back_to_host

                # Whether the launch allocates new memory beforehand or uses previous memory
                attr_accessor :reuse_memory

                # These fields can be read after building the kernel
                attr_reader :host_result_var_name

                attr_reader :kernel_result_var_name

                def initialize(kernel_builder)
                    @kernel_builder = kernel_builder
                    @additional_arguments = []
                    @previous_kernel_input = []
                    @write_back_to_host = false
                    @reuse_memory = false
                    @kernel_result_var_name = "_kernel_result_" + CommandTranslator.next_unique_id.to_s
                end

                def write_back_to_host!
                    @write_back_to_host = true
                    @host_result_var_name = @kernel_result_var_name + "_host"
                end

                def write_back_to_host?
                    return @write_back_to_host
                end

                def reuse_memory!(parameter_name)
                    @reuse_memory = true
                    @kernel_result_var_name = parameter_name
                    @host_result_var_name = @kernel_result_var_name + "_host"
                end

                def reuse_memory?
                    return @reuse_memory
                end

                def add_previous_kernel_parameter(parameter)
                    kernel_builder.add_previous_kernel_parameter(parameter)
                end

                def add_additional_arguments(argument)
                    @additional_arguments.push(argument)
                end

                # Configures grid size and block size. Also sets number of threads.
                def configure_grid(size)
                    @grid_dim = [size.fdiv(256).ceil, 1].max.to_s
                    @block_dim = (size >= 256 ? 256 : size).to_s
                    @num_threads = size
                end

                def assert_ready_to_build
                    required_values = [:num_threads, :grid_dim, :block_dim]

                    for selector in required_values
                        if send(selector) == nil
                            raise "Not ready to build (KernelBuilder): #{selector} is not set"
                        end
                    end
                end


                # Build the code that launches this kernel. The generated code performs the
                # following steps:
                #
                # 1. Allocate device memory for the result.
                # 2. If result should be written back: Allocate host memory for the result.
                # 3. Launch the kernel (+ error checking, synchronization)
                # 4. If result should be written back: Copy result back to host memory.
                def build_kernel_launcher
                    
                    Log.info("Building kernel launcher (write_back=#{write_back_to_host?})")

                    assert_ready_to_build

                    result = ""
                    if !reuse_memory
                        # Allocate device memory for kernel result
                        result = result + Translator.read_file(file_name: "allocate_device_memory.cpp", replacements: {
                            "name" => kernel_result_var_name,
                            "bytes" => "(#{kernel_builder.result_type.c_size} * #{num_threads})",
                            "type" => kernel_builder.result_type.to_c_type})
                    end

                    if write_back_to_host?
                        # Allocate host memory for kernel result
                        result = result + Translator.read_file(file_name: "allocate_host_memory.cpp", replacements: {
                            "name" => @host_result_var_name,
                            "bytes" => "(#{kernel_builder.result_type.c_size} * #{num_threads})",
                            "type" => kernel_builder.result_type.to_c_type})
                    end

                    # Build arguments
                    a_env = Constants::ENV_DEVICE_IDENTIFIER
                    a_result = kernel_result_var_name

                    previous_kernel_args = []
                    for var in kernel_builder.previous_kernel_input
                        previous_kernel_args.push(var.name.to_s)
                    end

                    if reuse_memory
                        previous_kernel_args[0] = a_result
                    end

                    arguments = ([a_env, num_threads, a_result] + previous_kernel_args + additional_arguments).join(", ")

                    # Launch kernel
                    result = result + Translator.read_file(file_name: "launch_kernel.cpp", replacements: {
                        "kernel_name" => kernel_builder.kernel_name,
                        "arguments" => arguments,
                        "grid_dim" => grid_dim,
                        "block_dim" => block_dim})

                    if write_back_to_host?
                        # Memcpy kernel result from device to host
                        result = result + Translator.read_file(file_name: "memcpy_device_to_host.cpp", replacements: {
                            "host_name" => @host_result_var_name,
                            "device_name" => @kernel_result_var_name,
                            "bytes" => "(#{kernel_builder.result_type.c_size} * #{num_threads})"})
                    end

                    return result
                end
                
                def build_device_memory_free
                    Log.info("Building kernel post-launch CUDA free")

                    assert_ready_to_build

                    return Translator.read_file(file_name: "free_device_memory.cpp", replacements: {
                        "name" => kernel_result_var_name})
                end
            end
        end
    end
end

require_relative "kernel_builder"