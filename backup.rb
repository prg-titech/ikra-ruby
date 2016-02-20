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
        
        
                        if var.is_index?
                    value = "threadIdx.x + blockIdx.x * blockDim.x"
                elsif var.is_normal?
                    value = "_input_#{var.name}[threadIdx.x + blockIdx.x * blockDim.x]"
                    command_proxy.add_input_type(var.type)
                    
                    launcher_params.push("#{var.type.to_c_type} *host_input_#{var.name}")
                    kernel_params.push("#{var.type.to_c_type} _input_#{var.name}")
                    kernel_args.push("device_input_#{var.name}")
                    device_input_decl += """#{var.type.to_c_type} *device_input_#{var.name};
    cudaMalloc(&device_input_#{var.name}, #{var.type.c_size} * #{size});
    cudaMemcpy(device_input_#{var.name}, host_input_#{var.name}, #{var.type.c_size} * #{size}, cudaMemcpyHostToDevice);
"""
                elsif var.is_fusion?
                    # TODO: handle multiple previous
                    value = "_previous_"
                end
                
                result = "#{var.type.to_c_type} #{var.name} = #{value};\n" + result