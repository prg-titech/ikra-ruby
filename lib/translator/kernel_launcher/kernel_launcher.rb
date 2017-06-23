module Ikra
    module Translator
        class CommandTranslator

            # Builds the launch of the kernel. This class is responsible for generating the 
            # invocation of the kernel.
            #
            # For example:
            # kernel<<<..., ...>>>(env, result, d_a, ...);
            class KernelLauncher
                class << self
                    # Debug flag only: Frees all input after launching kernel. This causes an
                    # error if data is used twice or kept (using the `keep` flag)
                    attr_accessor :debug_free_previous_input_immediately
                end

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

                # Whether the launch allocates new memory beforehand or uses previous memory
                attr_accessor :reuse_memory

                # Pointer to the resulting array (device memory)
                attr_reader :kernel_result_var_name

                # IDs and types of commands whose results are kept on the GPU
                attr_accessor :cached_results

                # IDs and types of commands that were previously computed and shall now be used in this kernel as input
                attr_reader :previously_cached_results

                def initialize(kernel_builder)
                    @kernel_builder = kernel_builder
                    @additional_arguments = []
                    @previous_kernel_input = []
                    @reuse_memory = false
                    @kernel_result_var_name = "_kernel_result_" + CommandTranslator.next_unique_id.to_s
                    @cached_results = {}
                    @previously_cached_results = {}
                end

                # Some of the values stored in `@additional_arguments` might be blocks, because
                # not all information was known when adding something to that list. This method
                # replaces those blocks (evaluates them) with actual strings, based on the command
                # that is being launched.
                def prepare_additional_args_for_launch(command)
                    @additional_arguments = @additional_arguments.map do |arg|
                        if arg.is_a?(String)
                            arg
                        else
                            arg.call(command)
                        end
                    end
                end

                def kernel_builders
                    # The program builder accesses kernel builders via kernel launchers through
                    # this method, because some specialized launchers might have multiple kernel 
                    # builders.
                    return [kernel_builder]
                end

                # Adds command whose result will be kept on GPU
                def add_cached_result(result_id, type)
                    @cached_results[result_id] = type
                end

                # Adds a previously computed result which will be used in this launche as input
                def use_cached_result(result_id, type)
                    @previously_cached_results[result_id] = type
                end

                def reuse_memory!(parameter_name)
                    @reuse_memory = true
                    @kernel_result_var_name = parameter_name
                end

                def reuse_memory?
                    return @reuse_memory
                end

                def add_previous_kernel_parameter(parameter)
                    kernel_builder.add_previous_kernel_parameter(parameter)
                end

                # Add additional arguments to the kernel function that might be needed for some computations
                def add_additional_arguments(*arguments)
                    @additional_arguments.push(*arguments)
                end

                # The result type of this kernel launcher. Same as the result type of its kernel 
                # builder.
                def result_type
                    return kernel_builder.result_type
                end

                # The size of the result array is the number of threads.
                def result_size
                    return num_threads
                end

                # Configures grid size and block size. Also sets number of threads.
                def configure_grid(size, block_size: 1024)
                    if block_size == nil
                        # TODO: How can this ever be nil? Should be set in symbolic.rb
                        block_size = 1024
                    end
                    
                    if size.is_a?(Fixnum)
                        # Precompute constants
                        @grid_dim = [size.fdiv(block_size).ceil, 1].max.to_s
                        @block_dim = (size >= block_size ? block_size : size).to_s
                        @num_threads = size
                    else
                        if !size.is_a?(String)
                            raise AssertionError.new("Fixnum or String expected")
                        end

                        # Source code string determines the size
                        @grid_dim = "max((int) ceil(((float) #{size}) / #{block_size}), 1)"
                        @block_dim = "(#{size} >= #{block_size} ? #{block_size} : #{size})"
                        @num_threads = size
                    end
                end

                def assert_ready_to_build
                    required_values = [:num_threads, :grid_dim, :block_dim]

                    for selector in required_values
                        if send(selector) == nil
                            raise AssertionError.new(
                                "Not ready to build (KernelBuilder): #{selector} is not set")
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
                    
                    Log.info("Building kernel launcher")

                    assert_ready_to_build

                    result = ""
                    if !reuse_memory
                        # Allocate device memory for kernel result
                        result = result + Translator.read_file(file_name: "allocate_device_memory.cpp", replacements: {
                            "name" => kernel_result_var_name,
                            "bytes" => "(sizeof(#{kernel_builder.result_type.to_c_type}) * #{num_threads})",
                            "type" => kernel_builder.result_type.to_c_type})
                    end

                    previously_cached_results.each do |result_id, type|
                        result = result + "    #{type.to_c_type} *prev_" + result_id.to_s + " = (#{type.to_c_type} *) " + Constants::ENV_HOST_IDENTIFIER + "->prev_" + result_id.to_s + ";\n"
                    end 

                    # Allocate device memory for cached results
                    cached_results.each do |result_id, type|
                        result = result + Translator.read_file(file_name: "allocate_device_memory.cpp", replacements: {
                            "name" => Constants::RESULT_IDENTIFIER + result_id,
                            "bytes" => "(#{type.c_size} * #{num_threads})",
                            "type" => type.to_c_type})
                    end

                    # Build arguments
                    a_env = Constants::ENV_DEVICE_IDENTIFIER
                    a_result = kernel_result_var_name

                    previous_kernel_args = []
                    for var in kernel_builder.previous_kernel_input
                        previous_kernel_args.push(var.name.to_s)
                    end

                    a_cached_results = cached_results.map do |result_id, type|
                        Constants::RESULT_IDENTIFIER + result_id
                    end

                    if reuse_memory
                        previous_kernel_args[0] = a_result
                    end

                    arguments = ([a_env, num_threads, a_result] + a_cached_results + previous_kernel_args + additional_arguments).join(", ")

                    # Launch kernel
                    result = result + Translator.read_file(file_name: "launch_kernel.cpp", replacements: {
                        "kernel_name" => kernel_builder.kernel_name,
                        "arguments" => arguments,
                        "grid_dim" => grid_dim,
                        "block_dim" => block_dim})

                    # ---- DEBUG ONLY: Free input after computation so that we can process larger
                    #                  data sets in benchmarks without running out of memory
                    # TODO: Implement analysis and do this automatically
                    if KernelLauncher.debug_free_previous_input_immediately == true
                        for var in kernel_builder.previous_kernel_input
                            result = result + Translator.read_file(file_name: "free_device_memory.cpp", replacements: {
                                "name" => var.name.to_s})
                        end 
                    end
                    # ---- END DEBUG ONLY

                    cached_results.each do |result_id, type|
                        result = result + "    " + Constants::ENV_HOST_IDENTIFIER + "->prev_" + result_id + " = " + Constants::RESULT_IDENTIFIER + result_id + ";\n"
                    end

                    return result
                end
                
                def build_device_memory_free
                    Log.info("Building kernel post-launch CUDA free")

                    assert_ready_to_build

                    if KernelLauncher.debug_free_previous_input_immediately == true
                        Log.warn("Debug flag set... Freeing input memory immediately and some memory not at all!")
                        return ""
                    end

                    return Translator.read_file(file_name: "free_device_memory.cpp", replacements: {
                        "name" => kernel_result_var_name})
                end

                # Same as above, but also removes item from the list of allocated memory chunks.
                def build_device_memory_free_in_host_section
                    Log.info("Building kernel post-launch CUDA free (host section")

                    assert_ready_to_build

                    return Translator.read_file(file_name: "host_section_free_device_memory.cpp", replacements: {
                        "name" => kernel_result_var_name})
                end
            end
        end
    end
end

require_relative "../kernel_builder"
require_relative "for_loop_kernel_launcher"
require_relative "while_loop_kernel_launcher"
