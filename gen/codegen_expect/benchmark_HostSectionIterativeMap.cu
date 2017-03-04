#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <vector>

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
    int *result;
    __host__ __device__ array_command_1(int *result = NULL) : result(result) { }
};
struct array_command_2 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_1 *input_0;
    __host__ __device__ array_command_2(int *result = NULL, array_command_1 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_3 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_2 *input_0;
    __host__ __device__ array_command_3(int *result = NULL, array_command_2 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_5 {
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    int *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_5(int *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_5 *input_0;
    __host__ __device__ array_command_6(int *result = NULL, array_command_5 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct environment_struct
{
};

// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}





// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_7(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_3_(_env_, _block_k_2_(_env_, _tid_));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_9(environment_t *_env_, int _num_threads_, int *_result_, int *_array_11_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_11_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_12(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}





// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_14(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_3_(_env_, _block_k_2_(_env_, _tid_));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_16(environment_t *_env_, int _num_threads_, int *_result_, int *_array_18_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_18_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_19(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}





// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_21(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_3_(_env_, _block_k_2_(_env_, _tid_));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_23(environment_t *_env_, int _num_threads_, int *_result_, int *_array_25_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_25_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_26(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}





// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_28(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_3_(_env_, _block_k_2_(_env_, _tid_));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_30(environment_t *_env_, int _num_threads_, int *_result_, int *_array_32_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_32_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_33(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}





// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_35(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_3_(_env_, _block_k_2_(_env_, _tid_));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_37(environment_t *_env_, int _num_threads_, int *_result_, int *_array_39_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_39_[_tid_]);
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
    array_command_2 * input = new array_command_2();
    int i;
    union_t _ssa_var_a_2;
    union_t _ssa_var_a_1;
    {
        _ssa_var_a_1 = union_t(10, union_v_t::from_pointer((void *) input));
        for (i = 1; i <= (100000 - 1); i++)
        {
            _ssa_var_a_2 = union_t(12, union_v_t::from_pointer((void *) new array_command_6(NULL, new array_command_5(NULL, ({
                variable_size_array_t _polytemp_result_1;
                {
                    union_t _polytemp_expr_2 = _ssa_var_a_1;
                    switch (_polytemp_expr_2.class_id)
                    {
                        case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 511]
                        
                            array_command_2 * cmd = (array_command_2 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_6;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 511)));
                            program_result->device_allocations->push_back(_kernel_result_6);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_5<<<2, 256>>>(dev_env, 511, _kernel_result_6);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_6;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 511);
                        }); break;
                        case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 511]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                        
                            array_command_3 * cmd = (array_command_3 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_8;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_8, (sizeof(int) * 511)));
                            program_result->device_allocations->push_back(_kernel_result_8);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_7<<<2, 256>>>(dev_env, 511, _kernel_result_8);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_8;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 511);
                        }); break;
                        case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 511]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                        
                            array_command_6 * cmd = (array_command_6 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_10;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_10, (sizeof(int) * 511)));
                            program_result->device_allocations->push_back(_kernel_result_10);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_9<<<2, 256>>>(dev_env, 511, _kernel_result_10, ((int *) cmd->input_0->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_10;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 511);
                        }); break;
                    }
                }
                _polytemp_result_1;
            })))));
            _ssa_var_a_1 = _ssa_var_a_2;
        }
        i--;
        return ({
            variable_size_array_t _polytemp_result_9;
            {
                union_t _polytemp_expr_10 = _ssa_var_a_1;
                switch (_polytemp_expr_10.class_id)
                {
                    case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 511]
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_34;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_34, (sizeof(int) * 511)));
                        program_result->device_allocations->push_back(_kernel_result_34);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_33<<<2, 256>>>(dev_env, 511, _kernel_result_34);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_34;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 511);
                    }); break;
                    case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 511]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_36;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_36, (sizeof(int) * 511)));
                        program_result->device_allocations->push_back(_kernel_result_36);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_35<<<2, 256>>>(dev_env, 511, _kernel_result_36);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_36;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 511);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 511] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 511]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                    
                        array_command_6 * cmd = (array_command_6 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_38;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_38, (sizeof(int) * 511)));
                        program_result->device_allocations->push_back(_kernel_result_38);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_37<<<2, 256>>>(dev_env, 511, _kernel_result_38, ((int *) cmd->input_0->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_38;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 511);
                    }); break;
                }
            }
            _polytemp_result_9;
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
