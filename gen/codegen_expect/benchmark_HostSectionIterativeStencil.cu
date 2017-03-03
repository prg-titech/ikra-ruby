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


/* ----- BEGIN Macros ----- */
#define timeStartMeasure() start_time = chrono::high_resolution_clock::now();

#define timeReportMeasure(result_var, variable_name) \
end_time = chrono::high_resolution_clock::now(); \
result_var->time_##variable_name = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
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
    // Ikra::Symbolic::ArrayReduceCommand
    int *result;
    array_command_2 *input_0;
    __host__ __device__ array_command_3(int *result = NULL, array_command_2 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayStencilCommand
    int *result;
    array_command_2 *input_0;
    __host__ __device__ array_command_4(int *result = NULL, array_command_2 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayReduceCommand
    int *result;
    array_command_4 *input_0;
    __host__ __device__ array_command_6(int *result = NULL, array_command_4 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayInHostSectionCommand
    int *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_5(int *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_11 {
    // Ikra::Symbolic::ArrayStencilCommand
    int *result;
    array_command_5 *input_0;
    __host__ __device__ array_command_11(int *result = NULL, array_command_5 *input_0 = NULL) : result(result), input_0(input_0) { }
    int size() { return input_0->size(); }
};
struct array_command_14 {
    // Ikra::Symbolic::ArrayReduceCommand
    int *result;
    array_command_11 *input_0;
    __host__ __device__ array_command_14(int *result = NULL, array_command_11 *input_0 = NULL) : result(result), input_0(input_0) { }
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_42(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_40(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_43, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_43[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_43[_tid_], _kernel_result_43[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_43[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_43[_tid_], _kernel_result_43[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_3_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_3_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_49(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_47(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_50)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_51;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_51 = _block_k_4_(_env_, _kernel_result_50[(temp_stencil_dim_0 + -1) * 1], _kernel_result_50[(temp_stencil_dim_0 + 0) * 1], _kernel_result_50[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_51 = 1;
    }
        
        _result_[_tid_] = temp_stencil_51;
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_45(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_48, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_48[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_48[_tid_], _kernel_result_48[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_48[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_48[_tid_], _kernel_result_48[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_6_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_6_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}




__global__ void kernel_57(environment_t *_env_, int _num_threads_, int *_result_, int *_array_59_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_59_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_55(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_58, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_60;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_60 = _block_k_11_(_env_, _kernel_result_58[(temp_stencil_dim_0 + -1) * 1], _kernel_result_58[(temp_stencil_dim_0 + 0) * 1], _kernel_result_58[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_60 = 1;
    }
        
        _result_[_tid_] = temp_stencil_60;
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_53(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_56, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_56[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_56[_tid_], _kernel_result_56[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_56[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_56[_tid_], _kernel_result_56[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_14_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_14_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_65(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_63(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_66, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_66[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_66[_tid_], _kernel_result_66[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_66[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_66[_tid_], _kernel_result_66[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_3_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_3_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_72(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_70(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_73)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_74;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_74 = _block_k_4_(_env_, _kernel_result_73[(temp_stencil_dim_0 + -1) * 1], _kernel_result_73[(temp_stencil_dim_0 + 0) * 1], _kernel_result_73[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_74 = 1;
    }
        
        _result_[_tid_] = temp_stencil_74;
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_68(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_71, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_71[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_71[_tid_], _kernel_result_71[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_71[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_71[_tid_], _kernel_result_71[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_6_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_6_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}




__global__ void kernel_80(environment_t *_env_, int _num_threads_, int *_result_, int *_array_82_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_82_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_78(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_81, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_83;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_83 = _block_k_11_(_env_, _kernel_result_81[(temp_stencil_dim_0 + -1) * 1], _kernel_result_81[(temp_stencil_dim_0 + 0) * 1], _kernel_result_81[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_83 = 1;
    }
        
        _result_[_tid_] = temp_stencil_83;
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_76(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_79, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_79[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_79[_tid_], _kernel_result_79[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_79[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_79[_tid_], _kernel_result_79[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_14_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_14_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_88(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_86(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_89, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_89[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_89[_tid_], _kernel_result_89[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_89[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_89[_tid_], _kernel_result_89[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_3_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_3_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_95(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_93(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_96)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_97;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_97 = _block_k_4_(_env_, _kernel_result_96[(temp_stencil_dim_0 + -1) * 1], _kernel_result_96[(temp_stencil_dim_0 + 0) * 1], _kernel_result_96[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_97 = 1;
    }
        
        _result_[_tid_] = temp_stencil_97;
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_91(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_94, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_94[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_94[_tid_], _kernel_result_94[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_94[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_94[_tid_], _kernel_result_94[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_6_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_6_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}




__global__ void kernel_103(environment_t *_env_, int _num_threads_, int *_result_, int *_array_105_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_105_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_101(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_104, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_106;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_106 = _block_k_11_(_env_, _kernel_result_104[(temp_stencil_dim_0 + -1) * 1], _kernel_result_104[(temp_stencil_dim_0 + 0) * 1], _kernel_result_104[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_106 = 1;
    }
        
        _result_[_tid_] = temp_stencil_106;
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_99(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_102, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_102[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_102[_tid_], _kernel_result_102[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_102[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_102[_tid_], _kernel_result_102[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_14_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_14_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_111(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_109(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_112, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_112[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_112[_tid_], _kernel_result_112[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_112[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_112[_tid_], _kernel_result_112[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_3_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_3_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_118(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_116(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_119)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_120;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_120 = _block_k_4_(_env_, _kernel_result_119[(temp_stencil_dim_0 + -1) * 1], _kernel_result_119[(temp_stencil_dim_0 + 0) * 1], _kernel_result_119[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_120 = 1;
    }
        
        _result_[_tid_] = temp_stencil_120;
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_114(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_117, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_117[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_117[_tid_], _kernel_result_117[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_117[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_117[_tid_], _kernel_result_117[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_6_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_6_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}




__global__ void kernel_126(environment_t *_env_, int _num_threads_, int *_result_, int *_array_128_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_128_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_124(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_127, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_129;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_129 = _block_k_11_(_env_, _kernel_result_127[(temp_stencil_dim_0 + -1) * 1], _kernel_result_127[(temp_stencil_dim_0 + 0) * 1], _kernel_result_127[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_129 = 1;
    }
        
        _result_[_tid_] = temp_stencil_129;
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int a, int b)
{
    
    
    {
        return (a + b);
    }
}

#endif


__global__ void kernel_122(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_125, bool _odd_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
        int thread_idx = threadIdx.x;

        // Single result of this block
        int _temp_result_;

        int num_args = 2 * 256;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * _num_threads_ - 1) % (2 * 256)) + (_odd_ ? 0 : 1);
        }

        if (num_args == 1)
        {
            _temp_result_ = _kernel_result_125[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_125[_tid_], _kernel_result_125[_tid_ + _num_threads_]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ int sdata[256];

            _odd_ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && _odd_)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = _kernel_result_125[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_125[_tid_], _kernel_result_125[_tid_ + _num_threads_]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            _odd_ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !_odd_)
                    {
                        sdata[thread_idx] = _block_k_14_(_env_, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                _odd_ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                _temp_result_ = _block_k_14_(_env_, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }
        
        _result_[_tid_] = _temp_result_;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_132(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_136(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_134(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_137)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_138;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_138 = _block_k_4_(_env_, _kernel_result_137[(temp_stencil_dim_0 + -1) * 1], _kernel_result_137[(temp_stencil_dim_0 + 0) * 1], _kernel_result_137[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_138 = 1;
    }
        
        _result_[_tid_] = temp_stencil_138;
    }
}




__global__ void kernel_141(environment_t *_env_, int _num_threads_, int *_result_, int *_array_143_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_143_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_139(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_142, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_144;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_144 = _block_k_11_(_env_, _kernel_result_142[(temp_stencil_dim_0 + -1) * 1], _kernel_result_142[(temp_stencil_dim_0 + 0) * 1], _kernel_result_142[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_144 = 1;
    }
        
        _result_[_tid_] = temp_stencil_144;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_145(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_149(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_147(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_150)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_151;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_151 = _block_k_4_(_env_, _kernel_result_150[(temp_stencil_dim_0 + -1) * 1], _kernel_result_150[(temp_stencil_dim_0 + 0) * 1], _kernel_result_150[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_151 = 1;
    }
        
        _result_[_tid_] = temp_stencil_151;
    }
}




__global__ void kernel_154(environment_t *_env_, int _num_threads_, int *_result_, int *_array_156_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_156_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_152(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_155, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_157;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_157 = _block_k_11_(_env_, _kernel_result_155[(temp_stencil_dim_0 + -1) * 1], _kernel_result_155[(temp_stencil_dim_0 + 0) * 1], _kernel_result_155[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_157 = 1;
    }
        
        _result_[_tid_] = temp_stencil_157;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_158(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_162(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_160(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_163)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_164;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_164 = _block_k_4_(_env_, _kernel_result_163[(temp_stencil_dim_0 + -1) * 1], _kernel_result_163[(temp_stencil_dim_0 + 0) * 1], _kernel_result_163[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_164 = 1;
    }
        
        _result_[_tid_] = temp_stencil_164;
    }
}




__global__ void kernel_167(environment_t *_env_, int _num_threads_, int *_result_, int *_array_169_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_169_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_165(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_168, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_170;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_170 = _block_k_11_(_env_, _kernel_result_168[(temp_stencil_dim_0 + -1) * 1], _kernel_result_168[(temp_stencil_dim_0 + 0) * 1], _kernel_result_168[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_170 = 1;
    }
        
        _result_[_tid_] = temp_stencil_170;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_171(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_175(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_173(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_176)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_177;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_177 = _block_k_4_(_env_, _kernel_result_176[(temp_stencil_dim_0 + -1) * 1], _kernel_result_176[(temp_stencil_dim_0 + 0) * 1], _kernel_result_176[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_177 = 1;
    }
        
        _result_[_tid_] = temp_stencil_177;
    }
}




__global__ void kernel_180(environment_t *_env_, int _num_threads_, int *_result_, int *_array_182_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_182_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_178(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_181, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_183;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_183 = _block_k_11_(_env_, _kernel_result_181[(temp_stencil_dim_0 + -1) * 1], _kernel_result_181[(temp_stencil_dim_0 + 0) * 1], _kernel_result_181[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_183 = 1;
    }
        
        _result_[_tid_] = temp_stencil_183;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j % 2);
    }
}

#endif


__global__ void kernel_184(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (j % 2);
    }
}

#endif


__global__ void kernel_188(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_186(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_189)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_190;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_190 = _block_k_4_(_env_, _kernel_result_189[(temp_stencil_dim_0 + -1) * 1], _kernel_result_189[(temp_stencil_dim_0 + 0) * 1], _kernel_result_189[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_190 = 1;
    }
        
        _result_[_tid_] = temp_stencil_190;
    }
}




__global__ void kernel_193(environment_t *_env_, int _num_threads_, int *_result_, int *_array_195_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_195_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2 };
    {
        return (((((values[0] - values[1])) - values[2])) + 7);
    }
}

#endif


__global__ void kernel_191(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_194, int dim_size_0)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_196;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < dim_size_0)
    {
        // All value indices within bounds
        
        temp_stencil_196 = _block_k_11_(_env_, _kernel_result_194[(temp_stencil_dim_0 + -1) * 1], _kernel_result_194[(temp_stencil_dim_0 + 0) * 1], _kernel_result_194[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_196 = 1;
    }
        
        _result_[_tid_] = temp_stencil_196;
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
    union_t _ssa_var_a_2;
    union_t _ssa_var_a_1;
    {
        _ssa_var_a_1 = union_t(13, union_v_t::from_pointer((void *) input));
        while (((((int *) ({
            variable_size_array_t device_array = ({
            variable_size_array_t _polytemp_result_11;
            {
                union_t _polytemp_expr_12 = ({
                    union_t _polytemp_result_13;
                    {
                        union_t _polytemp_expr_14 = _ssa_var_a_1;
                        switch (_polytemp_expr_14.class_id)
                        {
                            case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_13 = union_t(14, union_v_t::from_pointer((void *) new array_command_3(NULL, (array_command_2 *) _polytemp_expr_14.value.pointer))); break;
                            case 15: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_13 = union_t(16, union_v_t::from_pointer((void *) new array_command_6(NULL, (array_command_4 *) _polytemp_expr_14.value.pointer))); break;
                            case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = cmd->input_0->size()] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_13 = union_t(18, union_v_t::from_pointer((void *) new array_command_14(NULL, (array_command_11 *) _polytemp_expr_14.value.pointer))); break;
                        }
                    }
                    _polytemp_result_13;
                });
                switch (_polytemp_expr_12.class_id)
                {
                    case 14: /* [Ikra::Symbolic::ArrayReduceCommand, size = 1] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayReduceCommand, size = 1]: [SendNode: [LVarReadNode: _ssa_var_a_1].preduce()]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_43;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_43, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_43);
                        kernel_42<<<353, 256>>>(dev_env, 90210, _kernel_result_43);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_44;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_44, (sizeof(int) * 45105)));
                        program_result->device_allocations->push_back(_kernel_result_44);
                        kernel_40<<<177, 256>>>(dev_env, 45105, _kernel_result_44, _kernel_result_43, false);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        kernel_40<<<1, 89>>>(dev_env, 89, _kernel_result_44, _kernel_result_44, true);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_44;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                    case 16: /* [Ikra::Symbolic::ArrayReduceCommand, size = 1] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayReduceCommand, size = 1]: [SendNode: [LVarReadNode: _ssa_var_a_1].preduce()]
                    
                        array_command_6 * cmd = (array_command_6 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_50;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_50, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_50);
                        kernel_49<<<353, 256>>>(dev_env, 90210, _kernel_result_50);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_48;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_48, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_48);
                        kernel_47<<<353, 256>>>(dev_env, 90210, _kernel_result_48, _kernel_result_50);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_52;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_52, (sizeof(int) * 45105)));
                        program_result->device_allocations->push_back(_kernel_result_52);
                        kernel_45<<<177, 256>>>(dev_env, 45105, _kernel_result_52, _kernel_result_48, false);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        kernel_45<<<1, 89>>>(dev_env, 89, _kernel_result_52, _kernel_result_52, true);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_52;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                    case 18: /* [Ikra::Symbolic::ArrayReduceCommand, size = 1] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayReduceCommand, size = 1]: [SendNode: [LVarReadNode: _ssa_var_a_1].preduce()]
                    
                        array_command_14 * cmd = (array_command_14 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_58;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_58, (sizeof(int) * cmd->input_0->size())));
                        program_result->device_allocations->push_back(_kernel_result_58);
                        kernel_57<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_58, ((int *) ((int *) cmd->input_0->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_56;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_56, (sizeof(int) * cmd->input_0->size())));
                        program_result->device_allocations->push_back(_kernel_result_56);
                        kernel_55<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_56, _kernel_result_58, cmd->input_0->size());
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_61;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_61, (sizeof(int) * ((int) ceil(cmd->input_0->size() / 2.0)))));
                        program_result->device_allocations->push_back(_kernel_result_61);
                        kernel_53<<<max((int) ceil(((float) ((int) ceil(cmd->input_0->size() / 2.0))) / 256), 1), (((int) ceil(cmd->input_0->size() / 2.0)) >= 256 ? 256 : ((int) ceil(cmd->input_0->size() / 2.0)))>>>(dev_env, ((int) ceil(cmd->input_0->size() / 2.0)), _kernel_result_61, _kernel_result_56, (cmd->input_0->size() % 2 == 1));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    int _num_elements = ceil(((int) ceil(cmd->input_0->size() / 2.0)) / (double) 256);
                    bool _next_odd = _num_elements % 2 == 1;
                    int _next_threads = ceil(_num_elements / 2.0);
                    
                    while (_num_elements > 1) {
                        kernel_53<<<max((int) ceil(((float) _next_threads) / 256), 1), (_next_threads >= 256 ? 256 : _next_threads)>>>(dev_env, _next_threads, _kernel_result_61, _kernel_result_61, _next_odd);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                    _num_elements = ceil(_next_threads / (double) 256);
                    bool _next_odd = _num_elements % 2 == 0;
                    _next_threads = ceil(_num_elements / 2.0);
                    
                    }
                    
                            cmd->result = _kernel_result_61;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                }
            }
            _polytemp_result_11;
        });
            int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);
            checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
            variable_size_array_t((void *) tmp_result, device_array.size);
        }).content)[0] < 10000000)))
        {
            _ssa_var_a_2 = union_t(17, union_v_t::from_pointer((void *) new array_command_11(NULL, new array_command_5(NULL, ({
                variable_size_array_t _polytemp_result_27;
                {
                    union_t _polytemp_expr_28 = _ssa_var_a_1;
                    switch (_polytemp_expr_28.class_id)
                    {
                        case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 90210]
                        
                            array_command_2 * cmd = (array_command_2 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    int * _kernel_result_133;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_133, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_133);
                            kernel_132<<<353, 256>>>(dev_env, 90210, _kernel_result_133);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                        
                        
                                cmd->result = _kernel_result_133;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 90210);
                        }); break;
                        case 15: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    int * _kernel_result_137;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_137, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_137);
                            kernel_136<<<353, 256>>>(dev_env, 90210, _kernel_result_137);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                        
                            int * _kernel_result_135;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_135, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_135);
                            kernel_134<<<353, 256>>>(dev_env, 90210, _kernel_result_135, _kernel_result_137);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                        
                        
                                cmd->result = _kernel_result_135;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 90210);
                        }); break;
                        case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = cmd->input_0->size()] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = cmd->input_0->size()]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                        
                            array_command_11 * cmd = (array_command_11 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    int * _kernel_result_142;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_142, (sizeof(int) * cmd->input_0->size())));
                            program_result->device_allocations->push_back(_kernel_result_142);
                            kernel_141<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_142, ((int *) cmd->input_0->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                        
                            int * _kernel_result_140;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_140, (sizeof(int) * cmd->input_0->size())));
                            program_result->device_allocations->push_back(_kernel_result_140);
                            kernel_139<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_140, _kernel_result_142, cmd->input_0->size());
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                        
                        
                                cmd->result = _kernel_result_140;
                            }
                        
                            variable_size_array_t((void *) cmd->result, cmd->size());
                        }); break;
                    }
                }
                _polytemp_result_27;
            })))));
            _ssa_var_a_1 = _ssa_var_a_2;
        }
        return ({
            variable_size_array_t _polytemp_result_35;
            {
                union_t _polytemp_expr_36 = _ssa_var_a_1;
                switch (_polytemp_expr_36.class_id)
                {
                    case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_35 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 90210]
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_36.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_185;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_185, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_185);
                        kernel_184<<<353, 256>>>(dev_env, 90210, _kernel_result_185);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_185;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 90210);
                    }); break;
                    case 15: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_35 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                    
                        array_command_4 * cmd = (array_command_4 *) _polytemp_expr_36.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_189;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_189, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_189);
                        kernel_188<<<353, 256>>>(dev_env, 90210, _kernel_result_189);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_187;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_187, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_187);
                        kernel_186<<<353, 256>>>(dev_env, 90210, _kernel_result_187, _kernel_result_189);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_187;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 90210);
                    }); break;
                    case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = cmd->input_0->size()] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_35 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = cmd->input_0->size()]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                    
                        array_command_11 * cmd = (array_command_11 *) _polytemp_expr_36.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_194;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_194, (sizeof(int) * cmd->input_0->size())));
                        program_result->device_allocations->push_back(_kernel_result_194);
                        kernel_193<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_194, ((int *) cmd->input_0->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                        int * _kernel_result_192;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_192, (sizeof(int) * cmd->input_0->size())));
                        program_result->device_allocations->push_back(_kernel_result_192);
                        kernel_191<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_192, _kernel_result_194, cmd->input_0->size());
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_192;
                        }
                    
                        variable_size_array_t((void *) cmd->result, cmd->size());
                    }); break;
                }
            }
            _polytemp_result_35;
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
    // Variables for measuring time
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;

    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));
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
    timeStartMeasure();
        /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkErrorReturn(program_result, cudaMalloc(&dev_env, sizeof(environment_t)));
    checkErrorReturn(program_result, cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));

    timeReportMeasure(program_result, prepare_env);


    /* Copy back memory and set pointer of result */
    program_result->result = ({
    variable_size_array_t device_array = _host_section__(host_env, dev_env, program_result);
    int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
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
