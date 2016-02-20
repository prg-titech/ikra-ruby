        # generate kernel launcher
        launcher = """#include <stdio.h>
        
extern \"C\" __declspec(dllexport) #{result_type.to_c_type} *launch_kernel(#{launcher_params.join(", ")})
{
    printf(\"kernel launched\\n\");
    void *device_parameters;
    cudaMalloc(&device_parameters, #{lexical_size});
    cudaMemcpy(device_parameters, host_parameters, #{lexical_size}, cudaMemcpyHostToDevice);
    
    #{result_type.to_c_type} *host_result = (#{result_type.to_c_type} *) malloc(#{result_type.c_size} * #{size});
    #{result_type.to_c_type} *device_result;
    cudaMalloc(&device_result, #{result_type.c_size} * #{size});
    
    #{device_input_decl}
    
    dim3 dim_grid(#{dim3_grid[0]}, #{dim3_grid[1]}, #{dim3_grid[2]});
    dim3 dim_block(#{dim3_block[0]}, #{dim3_block[1]}, #{dim3_block[2]});
    
    kernel<<<dim_grid, dim_block>>>(#{kernel_args.join(", ")});
    
    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, #{result_type.c_size} * #{size}, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_parameters);
    
    return host_result;
}"""

        # TODO: check: threadIdx.x + blockIdx.x * blockDim.x < #{size}
        assign_result = "((#{result_type.to_c_type} *) #{ResultVariable})[threadIdx.x + blockIdx.x * blockDim.x] = #{TempResultVariable};\n"
        kernel_source = "__global__ void kernel(#{kernel_params.join(", ")})\n" + wrap_in_c_block("\#define _env_offset_ 0\n" + result + "\n" + assign_result)
        c_source = kernel_source + "\n\n" + launcher
        puts c_source
        
        command_proxy.c_source = c_source
        command_proxy.array_size = size
        command_proxy