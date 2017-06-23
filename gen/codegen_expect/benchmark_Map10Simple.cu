#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include <algorithm>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

using namespace std;


/* ----- BEGIN Shared Library Export ----- */
// taken from http://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(_GCC)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif
/* ----- END Shared Library Export ----- */

/* ----- BEGIN Class Type ----- */
typedef int obj_id_t;
typedef int class_id_t;
/* ----- END Class Type ----- */

/* ----- BEGIN Environment (lexical variables) ----- */
// environment_struct must be defined later
typedef struct environment_struct environment_t;
/* ----- END Environment (lexical variables) ----- */


/* ----- BEGIN Forward declarations ----- */
typedef struct result_t result_t;
/* ----- END Forward declarations ----- */

// Define program result variable. Also contains benchmark numbers.
result_t *program_result;

// Variables for measuring time
chrono::high_resolution_clock::time_point start_time;
chrono::high_resolution_clock::time_point end_time;

/* ----- BEGIN Macros ----- */
#define timeStartMeasure() start_time = chrono::high_resolution_clock::now();

#define timeReportMeasure(result_var, variable_name) \
end_time = chrono::high_resolution_clock::now(); \
result_var->time_##variable_name = result_var->time_##variable_name + chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
/* ----- END Macros ----- */
struct indexed_struct_4_lt_int_int_int_int_gt_t
{
    int field_0;
int field_1;
int field_2;
int field_3;
};

/* ----- BEGIN Structs ----- */
struct variable_size_array_t {
    void *content;
    int size;

    variable_size_array_t(void *content_ = NULL, int size_ = 0) : content(content_), size(size_) { }; 

    static const variable_size_array_t error_return_value;
};

// error_return_value is used in case a host section terminates abnormally
const variable_size_array_t variable_size_array_t::error_return_value = 
    variable_size_array_t(NULL, 0);

/* ----- BEGIN Union Type ----- */
typedef union union_type_value {
    obj_id_t object_id;
    int int_;
    float float_;
    bool bool_;
    void *pointer;
    variable_size_array_t variable_size_array;

    __host__ __device__ union_type_value(int value) : int_(value) { };
    __host__ __device__ union_type_value(float value) : float_(value) { };
    __host__ __device__ union_type_value(bool value) : bool_(value) { };
    __host__ __device__ union_type_value(void *value) : pointer(value) { };
    __host__ __device__ union_type_value(variable_size_array_t value) : variable_size_array(value) { };

    __host__ __device__ static union_type_value from_object_id(obj_id_t value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_int(int value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_float(float value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_bool(bool value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_pointer(void *value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_variable_size_array_t(variable_size_array_t value)
    {
        return union_type_value(value);
    }
} union_v_t;

typedef struct union_type_struct
{
    class_id_t class_id;
    union_v_t value;

    __host__ __device__ union_type_struct(
        class_id_t class_id_ = 0, union_v_t value_ = union_v_t(0))
        : class_id(class_id_), value(value_) { };

    static const union_type_struct error_return_value;
} union_t;

// error_return_value is used in case a host section terminates abnormally
const union_type_struct union_t::error_return_value = union_type_struct(0, union_v_t(0));
/* ----- END Union Type ----- */

typedef struct result_t {
    variable_size_array_t result;
    int last_error;

    uint64_t time_setup_cuda;
    uint64_t time_prepare_env;
    uint64_t time_kernel;
    uint64_t time_free_memory;
    uint64_t time_transfer_memory;
    uint64_t time_allocate_memory;

    // Memory management
    vector<void*> *device_allocations;
} result_t;
/* ----- END Structs ----- */

struct array_command_1 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_4_lt_int_int_int_int_gt_t *result;
    __host__ __device__ array_command_1(indexed_struct_4_lt_int_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_2 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_1 *input_0;
    __host__ __device__ array_command_2(int *result = NULL, array_command_1 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_4_lt_int_int_int_int_gt_t *result;
    __host__ __device__ array_command_4(indexed_struct_4_lt_int_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_3 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_2 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_3(int *result = NULL, array_command_2 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_5(int *result = NULL, array_command_3 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_7 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_5 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_7(int *result = NULL, array_command_5 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_9 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_7 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_9(int *result = NULL, array_command_7 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_11 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_9 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_11(int *result = NULL, array_command_9 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_13 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_11 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_13(int *result = NULL, array_command_11 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_15 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_13 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_15(int *result = NULL, array_command_13 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_17 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_15 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_17(int *result = NULL, array_command_15 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_19 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_17 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_19(int *result = NULL, array_command_17 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_21 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_19 *input_0;
    array_command_4 *input_1;
    __host__ __device__ array_command_21(int *result = NULL, array_command_19 *input_0 = NULL, array_command_4 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct environment_struct
{
};





















// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    {
        return (((((((((7 + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_7_ is already defined
#ifndef _block_k_7__func
#define _block_k_7__func
__device__ int _block_k_7_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_9_ is already defined
#ifndef _block_k_9__func
#define _block_k_9__func
__device__ int _block_k_9_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_13_ is already defined
#ifndef _block_k_13__func
#define _block_k_13__func
__device__ int _block_k_13_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_15_ is already defined
#ifndef _block_k_15__func
#define _block_k_15__func
__device__ int _block_k_15_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_21_ is already defined
#ifndef _block_k_21__func
#define _block_k_21__func
__device__ int _block_k_21_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((((i + indices.field_0)) + indices.field_1)) + indices.field_2)) + indices.field_3)) % 1337);
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_21_(_env_, _block_k_19_(_env_, _block_k_17_(_env_, _block_k_15_(_env_, _block_k_13_(_env_, _block_k_11_(_env_, _block_k_9_(_env_, _block_k_7_(_env_, _block_k_5_(_env_, _block_k_3_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
}


#undef checkErrorReturn
#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return variable_size_array_t::error_return_value;\
}

variable_size_array_t _host_section__(environment_t *host_env, environment_t *dev_env, result_t *program_result)
{
    array_command_2 * base = new array_command_2();
    array_command_21 * _ssa_var_base_10;
    array_command_19 * _ssa_var_base_9;
    array_command_17 * _ssa_var_base_8;
    array_command_15 * _ssa_var_base_7;
    array_command_13 * _ssa_var_base_6;
    array_command_11 * _ssa_var_base_5;
    array_command_9 * _ssa_var_base_4;
    array_command_7 * _ssa_var_base_3;
    array_command_5 * _ssa_var_base_2;
    array_command_3 * _ssa_var_base_1;
    {
        _ssa_var_base_1 = new array_command_3(NULL, base);
        _ssa_var_base_2 = new array_command_5(NULL, _ssa_var_base_1);
        _ssa_var_base_3 = new array_command_7(NULL, _ssa_var_base_2);
        _ssa_var_base_4 = new array_command_9(NULL, _ssa_var_base_3);
        _ssa_var_base_5 = new array_command_11(NULL, _ssa_var_base_4);
        _ssa_var_base_6 = new array_command_13(NULL, _ssa_var_base_5);
        _ssa_var_base_7 = new array_command_15(NULL, _ssa_var_base_6);
        _ssa_var_base_8 = new array_command_17(NULL, _ssa_var_base_7);
        _ssa_var_base_9 = new array_command_19(NULL, _ssa_var_base_8);
        _ssa_var_base_10 = new array_command_21(NULL, _ssa_var_base_9);
        return ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 60000000]: [SendNode: [LVarReadNode: _ssa_var_base_9].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_21 * cmd = _ssa_var_base_10;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_2;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(int) * 60000000)));
            program_result->device_allocations->push_back(_kernel_result_2);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_1<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_2);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_2;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 60000000);
        });
    }
}

#undef checkErrorReturn
#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return result_var;\
}

extern "C" EXPORT result_t *launch_kernel(environment_t *host_env)
{
    // CUDA Initialization
    program_result = new result_t();
    program_result->device_allocations = new vector<void*>();

    timeStartMeasure();

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        program_result->last_error = -1;
        return program_result;
    }

    checkErrorReturn(program_result, cudaFree(0));

    timeReportMeasure(program_result, setup_cuda);


    /* Prepare environment */
        /* Allocate device environment and copy over struct */
    environment_t *dev_env;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc(&dev_env, sizeof(environment_t)));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);
    


    /* Copy back memory and set pointer of result */
    program_result->result = ({
    variable_size_array_t device_array = _host_section__(host_env, dev_env, program_result);
    int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
    timeReportMeasure(program_result, transfer_memory);

    variable_size_array_t((void *) tmp_result, device_array.size);
});

    /* Free device memory */
    timeStartMeasure();

    for (
        auto device_ptr = program_result->device_allocations->begin(); 
        device_ptr < program_result->device_allocations->end(); 
        device_ptr++)
    {
        checkErrorReturn(program_result, cudaFree(*device_ptr));
    }

    delete program_result->device_allocations;

    timeReportMeasure(program_result, free_memory);

    return program_result;
}
