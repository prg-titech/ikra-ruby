require_relative "translator"
require_relative "block_translator"
require_relative "method_translator"
require_relative "../scope"
require "logger"
require "tempfile"

module IkraAA
    Log = Logger.new(STDOUT)

    module Translator
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
                    @previous_block_result_type = nil
                    @invocation_source = ""
                    @kernel_inner_source = ""
                    @kernel_params = ["struct #{EnvStructName} *_env_"]
                    @kernel_args = ["device_env"]
                    @launcher_input_decl = ""
                    @launcher_params = ["struct #{EnvStructName} *host_env"]
                    @env_builder = EnvironmentBuilder.new
                    @initial_size = nil
                    
                    # Methods
                    @aux_methods = []

                    # Variables needed for FFI wrapper
                    @ffi_wrapper = nil
                    @env_struct_type = nil
                    @expected_input_types = []
                    @input_accumulator = []
                end
                
                def merge_request!(request)
                    @initial_size ||= request.size
                    
                    # Fill in types from previous translation
                    num_fusion_request = request.input_vars.select do |var|
                        var.is_fusion?
                    end.size
                    if num_fusion_request > 1
                        raise "More than one fusion request"
                    end
                    if num_fusion_request == 1 and @previous_block_result_type == nil
                        raise "No previous result type provided"
                    end
                    
                    request.input_vars.each do |var|
                        if var.is_fusion?
                            var.type = @previous_block_result_type
                        end
                    end
                    
                    # Input types
                    input_types = request.input_vars.map do |var|
                        var.type
                    end

                    # Translate next block
                    block_result = Translator.translate_block(
                        block: request.block, 
                        input_types: input_types, 
                        env_builder: @env_builder)
                    
                    @kernel_inner_source += block_result.c_source
                    @aux_methods += block_result.aux_methods.values

                    # Call arguments/parameters
                    inner_kernel_args = [EnvironmentVariable]
                    arg_index = 0
                    request.input_vars.each do |var|
                        arg_string = nil
                        if var.is_fusion?
                            # TODO: handle multiple arguments
                            arg_string = @invocation_source
                        elsif var.is_index?
                            arg_string = "threadIdx.x + blockIdx.x * blockDim.x"
                        elsif var.is_normal?
                            @expected_input_types.push(var.type.singleton_type)
                            @launcher_params.push("#{var.type.singleton_type.to_c_type} *host_k#{@block_index}_#{arg_index}")
                            @kernel_params.push("#{var.type.singleton_type.to_c_type} *_input_k#{@block_index}_#{arg_index}_")
                            arg_string = "_input_k#{@block_index}_#{arg_index}_[threadIdx.x + blockIdx.x * blockDim.x]"
                            
                            @launcher_input_decl += """#{var.type.singleton_type.to_c_type} *device_k#{@block_index}_#{arg_index};
    cudaMalloc(&device_k#{@block_index}_#{arg_index}, #{var.type.singleton_type.c_size} * #{request.size});
    cudaMemcpy(device_k#{@block_index}_#{arg_index}, host_k#{@block_index}_#{arg_index}, #{var.type.singleton_type.c_size} * #{request.size}, cudaMemcpyHostToDevice);
"""
                            @kernel_args.push("device_k#{@block_index}_#{arg_index}")
                        end
                        inner_kernel_args.push(arg_string)
                        arg_index += 1
                    end
                    
                    @invocation_source = "#{block_result.function_name}(#{inner_kernel_args.join(", ")})"
                    
                    @previous_block_result_type = block_result.result_type
                    @block_index += 1
                end
                
                def full_source
                    dimensions = grid_block_size(@initial_size)
                    dim3_grid = dimensions[0]
                    dim3_block = dimensions[1]
                    
                    preamble = "#define objid_t int\n\n"

                    struct_def = @env_builder.struct_definition(EnvStructName)

                    result_type = @previous_block_result_type
                    kernel_params = ["#{result_type.singleton_type.to_c_type} *_result_"] + @kernel_params
                    kernel_args = ["device_result"] + @kernel_args
                    
                    # TODO: handle multiple result types
                    launcher = """#include <stdio.h>

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(_GCC)
    //  GCC
    #define EXPORT __attribute__((visibility(\"default\")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

extern \"C\" EXPORT #{result_type.singleton_type.to_c_type} *launch_kernel(#{@launcher_params.join(", ")})
{
    printf(\"kernel launched\\n\");
    struct #{EnvStructName} *device_env;
    cudaMalloc(&device_env, sizeof(struct #{EnvStructName}));
    cudaMemcpy(device_env, host_env, sizeof(struct #{EnvStructName}), cudaMemcpyHostToDevice);
    
    #{result_type.singleton_type.to_c_type} *host_result = (#{result_type.singleton_type.to_c_type} *) malloc(#{result_type.singleton_type.c_size} * #{@initial_size});
    #{result_type.singleton_type.to_c_type} *device_result;
    cudaMalloc(&device_result, #{result_type.singleton_type.c_size} * #{@initial_size});
    
    #{@launcher_input_decl}
    
    dim3 dim_grid(#{dim3_grid[0]}, #{dim3_grid[1]}, #{dim3_grid[2]});
    dim3 dim_block(#{dim3_block[0]}, #{dim3_block[1]}, #{dim3_block[2]});
    
    kernel<<<dim_grid, dim_block>>>(#{kernel_args.join(", ")});
    
    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, #{result_type.singleton_type.c_size} * #{@initial_size}, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_env);
    // TODO: free input arg device mem
    
    return host_result;
}"""

                aux_methods_source = @aux_methods.map do |meth|
                    meth.to_c_source
                end.join("\n")

                kernel_source = """__global__ void kernel(#{kernel_params.join(", ")})
{
    _result_[threadIdx.x + blockIdx.x * blockDim.x] = #{@invocation_source};
}
"""

                    preamble + struct_def + aux_methods_source + "\n" + @kernel_inner_source + "\n" + kernel_source + "\n" + launcher
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
                    if @expected_input_types[next_index] == Types::PrimitiveType::Int
                        arr.put_array_of_int(0, array)
                    elsif @expected_input_types[next_index] == Types::PrimitiveType::Float
                        arr.put_array_of_float(0, array)
                    else
                        raise "Cannot handle array of #{@result_type} via FFI"
                    end
                    @input_accumulator.push(arr)
                end
                
                def build
                    Log.info("Full CUDA source: \n#{full_source}")
                    
                    file = Tempfile.new(["ikra_kernel", ".cu"])
                    file.write(full_source)
                    file.close
                    
                    compiler_invocation = "nvcc -o #{file.path}.so --shared -Xcompiler -fPIC #{file.path}"
                    time_before = Time.now
                    Log.info("Compiling CUDA code: #{compiler_invocation}")
                    compile_status = %x(#{compiler_invocation})
                    Log.info("Done, took #{Time.now - time_before} s")
                    
                    # Generate env struct
                    @env_struct_type = @env_builder.to_ffi_struct_type
                    
                    @ffi_wrapper = Module.new
                    @ffi_wrapper.extend(FFI::Library)
                    @ffi_wrapper.ffi_lib(file.path + ".so")
                    @ffi_wrapper.attach_function(:launch_kernel, [:pointer] * (@expected_input_types.size + 1), :pointer)
                end
                
                def execute
                    build
                    
                    if @input_accumulator.size != @expected_input_types.size
                        raise "Expected #{@expected_input_types.size} parameters (#{@input_accumulator.size} given)"
                    end
                    
                    all_input = []
                    
                    if @env_builder.size > 0
                        env_struct = @env_struct_type.new
                        (0..(@env_builder.size - 1)).each do |index|
                            env_struct[:"field_#{index}"] = @env_builder.get_var(index)
                        end
                        
                        all_input.push(env_struct.to_ptr)
                    else
                        # Empty struct: pass some pointer
                        all_input.push(FFI::MemoryPointer.new(0))
                    end
                    
                    all_input += @input_accumulator
                    
                    # Measure time and launch kernel
                    Log.info("Launching kernel...")
                    time_before = Time.now
                    result = @ffi_wrapper.launch_kernel(*all_input)
                    Log.info("Kernel time: #{Time.now - time_before} s")
                    
                    all_input.each do |pointer|
                        pointer.free
                    end
                    
                    # TODO: handle multiple return values
                    # TODO: handle multiple types of each return value
                    result_type = @previous_block_result_type.singleton_type
                    return_value = nil
                    if result_type == Types::PrimitiveType::Int
                        return_value = result.read_array_of_int(@initial_size)
                    elsif result_type == Types::PrimitiveType::Float
                        return_value = result.read_array_of_float(@initial_size)
                    else
                        raise "Cannot retrieve array of #{result_type} via FFI"
                    end
                    
                    Log.info("Return values FFI transfer complete")
                    return_value
                end
            end
            
            def self.compile(request)
                compilation_result = KernelCompilationResult.new
                compilation_result.merge_request!(request)
                compilation_result
            end
        end
    end
end