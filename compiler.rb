require_relative "translator"

module Compiler
    class InputVariable
        Normal = 0
        Index = 1
        PreviousFusion = 2
        
        attr_accessor :type
        
        def initialize(name, type, source_type = Normal)
            @name = name
            @type = type
            @source_type = source_type
        end
        
        def name
            @name
        end
        
        def is_index?
            @source_type == Index
        end
        
        def is_fusion?
            @source_type == PreviousFusion
        end
        
        def is_normal?
            @source_type == Normal
        end
    end
    
    class CompilationRequest
        attr_accessor :block
        attr_accessor :size
        
        # Array of InputVariable
        attr_reader :input_vars
        
        def initialize(block:, size:)
            @block = block
            @size = size
            @input_vars = []
        end

        def add_input_var(var)
            @input_vars.push(var)
        end
    end
    
    class KernelCompilationResult
        attr_accessor :c_source
        
        EnvironmentVariable = "_env_"
        
        def initialize
            # Variables used during compilation
            @block_index = 0
            @previous_block_result_types = []
            @invocation_source = ""
            @kernel_inner_source = ""
            @kernel_params = ["char *_env_"]
            @kernel_args = ["device_env"]
            @launcher_input_decl = ""
            @launcher_params = ["char *host_env"]
            @env_offset = 0
            @env_size = 0
            @initial_size = nil
            
            # Variables needed for FFI wrapper
            @ffi_wrapper = nil
            @expected_input_types = []
            @input_accumulator = []
            @env_vars = []
        end
        
        def merge_request!(request)
            @initial_size ||= request.size
            
            # Fill in types from previous translation
            num_fusion_request = request.input_vars.select do |var|
                var.is_fusion?
            end.size
            if num_fusion_request != @previous_block_result_types.size
                raise "Mismatch in number of resulting values (expected #{num_fusion_request}, found #{@previous_block_result_types.size}"
            end
            
            previous_type_index = 0
            request.input_vars.each do |var|
                if var.is_fusion?
                    var.type = @previous_block_result_types[previous_type_index]
                    previous_type_index += 1
                end
            end
            
            # Translate next block
            block_result = Translator.translate_block(
                block: request.block, 
                size: request.size, 
                input_vars: request.input_vars, 
                function_name: "kernel_inner_#{@block_index}")
            
            @kernel_inner_source += block_result.c_source
            
            # Call arguments/parameters
            inner_kernel_args = ["#{EnvironmentVariable} + #{@env_offset}"]
            arg_index = 0
            request.input_vars.each do |var|
                arg_string = nil
                if var.is_fusion?
                    # TODO: handle multiple arguments
                    arg_string = @invocation_source
                elsif var.is_index?
                    arg_string = "threadIdx.x + blockIdx.x * blockDim.x"
                elsif var.is_normal?
                    @expected_input_types.push(var.type)
                    @launcher_params.push("#{var.type.to_c_type} *host_k#{@block_index}_#{arg_index}")
                    @kernel_params.push("#{var.type.to_c_type} *_input_k#{@block_index}_#{arg_index}_")
                    arg_string = "_input_k#{@block_index}_#{arg_index}_[threadIdx.x + blockIdx.x * blockDim.x]"
                    
                    @launcher_input_decl += """#{var.type.to_c_type} *device_k#{@block_index}_#{arg_index};
    cudaMalloc(&device_k#{@block_index}_#{arg_index}, #{var.type.c_size} * #{request.size});
    cudaMemcpy(device_k#{@block_index}_#{arg_index}, host_k#{@block_index}_#{arg_index}, #{var.type.c_size} * #{request.size}, cudaMemcpyHostToDevice);
"""
                    @kernel_args.push("device_k#{@block_index}_#{arg_index}")
                end
                inner_kernel_args.push(arg_string)
                arg_index += 1
            end
            
            # Merge environment variables
            block_result.env_vars.each do |var|
                @env_vars.push(Translator::EnvironmentVariable.new(accessor: var.accessor, type: var.type, offset: var.offset + @env_offset))
            end
            
            @invocation_source = "kernel_inner_#{@block_index}(#{inner_kernel_args.join(", ")})"
            
            @previous_block_result_types = block_result.result_type
            @env_offset += block_result.env_size
            @env_size += block_result.env_size
            
            @block_index += 1
        end
        
        def full_source
            dimensions = grid_block_size(@initial_size)
            dim3_grid = dimensions[0]
            dim3_block = dimensions[1]
            
            result_type = @previous_block_result_types.first
            kernel_params = ["#{result_type.to_c_type} *_result_"] + @kernel_params
            kernel_args = ["device_result"] + @kernel_args
            
            # TODO: handle multiple result types
            launcher = """#include <stdio.h>

extern \"C\" __declspec(dllexport) #{result_type.to_c_type} *launch_kernel(#{@launcher_params.join(", ")})
{
    printf(\"kernel launched\\n\");
    char *device_env;
    cudaMalloc(&device_env, #{@env_size});
    cudaMemcpy(device_env, host_env, #{@env_size}, cudaMemcpyHostToDevice);
    
    #{result_type.to_c_type} *host_result = (#{result_type.to_c_type} *) malloc(#{result_type.c_size} * #{@initial_size});
    #{result_type.to_c_type} *device_result;
    cudaMalloc(&device_result, #{result_type.c_size} * #{@initial_size});
    
    #{@launcher_input_decl}
    
    dim3 dim_grid(#{dim3_grid[0]}, #{dim3_grid[1]}, #{dim3_grid[2]});
    dim3 dim_block(#{dim3_block[0]}, #{dim3_block[1]}, #{dim3_block[2]});
    
    kernel<<<dim_grid, dim_block>>>(#{kernel_args.join(", ")});
    
    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, #{result_type.c_size} * #{@initial_size}, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_env);
    // TODO: free input arg device mem
    
    return host_result;
}"""

        kernel_source = """__global__ void kernel(#{kernel_params.join(", ")})
{
    _result_[threadIdx.x + blockIdx.x * blockDim.x] = #{@invocation_source};
}
"""

            @kernel_inner_source + "\n" + kernel_source + "\n" + launcher
        end
        
        def grid_block_size(size)
            [[[size / 250, 1].max, 1, 1], [(size >= 250 ? 250 : size), 1, 1]]
        end
        
        def wrap_in_c_block(str)
            "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
        end
        
        def allocate_input_array(array)
            next_index = @input_accumulator.size
            
            if array.first.class.to_ikra_type != @expected_input_types[next_index]
                raise "Attempting to pass array of incompatible type #{array.first.class.to_ikra_type} (#{@expected_input_types[next_index]} expected)"
            end
            
            arr = FFI::MemoryPointer.new(array.size * @expected_input_types[next_index].c_size)
            if @expected_input_types[next_index] == PrimitiveType::Int
                arr.put_array_of_int(0, array)
            elsif @expected_input_types[next_index] == PrimitiveType::Float
                arr.put_array_of_float(0, array)
            else
                raise "Cannot handle array of #{@result_type} via FFI"
            end
            @input_accumulator.push(arr)
        end
        
        def build
            puts full_source
            
            file = Tempfile.new(["ikra_kernel", ".cu"])
            file.write(full_source)
            file.close
            
            compile_status = %x(nvcc -o #{file.path}.dll --shared #{file.path})
            
            @ffi_wrapper = Module.new
            @ffi_wrapper.extend(FFI::Library)
            @ffi_wrapper.ffi_lib(file.path + ".dll")
            @ffi_wrapper.attach_function(:launch_kernel, [:pointer] * (@expected_input_types.size + 1), :pointer)
        end
        
        def execute
            build
            
            if @input_accumulator.size != @expected_input_types.size
                raise "Expected #{@expected_input_types.size} parameters (#{@input_accumulator.size} given)"
            end
            
            env = FFI::MemoryPointer.new(@env_size)
            @env_vars.each do |var|
                value = var.accessor.call
                if value.class == Float
                    env.put_float(var.offset, value)
                elsif value.class == Fixnum
                    env.put_int(var.offset, value)
                else
                    raise "Cannot pass object of type #{value.class} via FFI"
                end
            end
            
            all_input = [env]
            all_input += @input_accumulator
            result = @ffi_wrapper.launch_kernel(*all_input)
            
            all_input.each do |pointer|
                pointer.free
            end
            
            # TODO: handle multiple return values
            result_type = @previous_block_result_types.first
            if result_type == PrimitiveType::Int
                result.read_array_of_int(@initial_size)
            elsif result_type == PrimitiveType::Float
                result.read_array_of_float(@initial_size)
            else
                raise "Cannot retrieve array of #{result_type} via FFI"
            end
        end
    end
    
    def self.compile(request)
        compilation_result = KernelCompilationResult.new
        compilation_result.merge_request!(request)
        compilation_result
    end
end