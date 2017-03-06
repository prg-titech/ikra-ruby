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
struct array_command_10 {
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    int *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_10(int *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_11 {
    // Ikra::Symbolic::ArrayStencilCommand
    int *result;
    array_command_10 *input_0;
    __host__ __device__ array_command_11(int *result = NULL, array_command_10 *input_0 = NULL) : result(result), input_0(input_0) { }
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


__global__ void kernel_55(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_58)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_60;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
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


__global__ void kernel_64(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_62(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_65, bool _odd_)
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
            _temp_result_ = _kernel_result_65[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_65[_tid_], _kernel_result_65[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_65[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_65[_tid_], _kernel_result_65[_tid_ + _num_threads_]);
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


__global__ void kernel_71(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_69(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_72)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_73;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_73 = _block_k_4_(_env_, _kernel_result_72[(temp_stencil_dim_0 + -1) * 1], _kernel_result_72[(temp_stencil_dim_0 + 0) * 1], _kernel_result_72[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_73 = 1;
    }
        
        _result_[_tid_] = temp_stencil_73;
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


__global__ void kernel_67(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_70, bool _odd_)
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
            _temp_result_ = _kernel_result_70[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_70[_tid_], _kernel_result_70[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_70[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_70[_tid_], _kernel_result_70[_tid_ + _num_threads_]);
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




__global__ void kernel_79(environment_t *_env_, int _num_threads_, int *_result_, int *_array_81_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_81_[_tid_];
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


__global__ void kernel_77(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_80)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_82;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_82 = _block_k_11_(_env_, _kernel_result_80[(temp_stencil_dim_0 + -1) * 1], _kernel_result_80[(temp_stencil_dim_0 + 0) * 1], _kernel_result_80[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_82 = 1;
    }
        
        _result_[_tid_] = temp_stencil_82;
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


__global__ void kernel_75(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_78, bool _odd_)
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
            _temp_result_ = _kernel_result_78[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_78[_tid_], _kernel_result_78[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_78[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_78[_tid_], _kernel_result_78[_tid_ + _num_threads_]);
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


__global__ void kernel_86(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_84(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_87, bool _odd_)
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
            _temp_result_ = _kernel_result_87[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_87[_tid_], _kernel_result_87[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_87[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_87[_tid_], _kernel_result_87[_tid_ + _num_threads_]);
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


__global__ void kernel_93(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_91(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_94)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_95;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_95 = _block_k_4_(_env_, _kernel_result_94[(temp_stencil_dim_0 + -1) * 1], _kernel_result_94[(temp_stencil_dim_0 + 0) * 1], _kernel_result_94[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_95 = 1;
    }
        
        _result_[_tid_] = temp_stencil_95;
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


__global__ void kernel_89(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_92, bool _odd_)
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
            _temp_result_ = _kernel_result_92[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_92[_tid_], _kernel_result_92[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_92[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_92[_tid_], _kernel_result_92[_tid_ + _num_threads_]);
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




__global__ void kernel_101(environment_t *_env_, int _num_threads_, int *_result_, int *_array_103_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_103_[_tid_];
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


__global__ void kernel_99(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_102)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_104;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_104 = _block_k_11_(_env_, _kernel_result_102[(temp_stencil_dim_0 + -1) * 1], _kernel_result_102[(temp_stencil_dim_0 + 0) * 1], _kernel_result_102[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_104 = 1;
    }
        
        _result_[_tid_] = temp_stencil_104;
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


__global__ void kernel_97(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_100, bool _odd_)
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
            _temp_result_ = _kernel_result_100[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_100[_tid_], _kernel_result_100[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_100[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_100[_tid_], _kernel_result_100[_tid_ + _num_threads_]);
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


__global__ void kernel_108(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_106(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_109, bool _odd_)
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
            _temp_result_ = _kernel_result_109[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_3_(_env_, _kernel_result_109[_tid_], _kernel_result_109[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_109[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_3_(_env_, _kernel_result_109[_tid_], _kernel_result_109[_tid_ + _num_threads_]);
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


__global__ void kernel_115(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_113(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_116)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_117;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_117 = _block_k_4_(_env_, _kernel_result_116[(temp_stencil_dim_0 + -1) * 1], _kernel_result_116[(temp_stencil_dim_0 + 0) * 1], _kernel_result_116[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_117 = 1;
    }
        
        _result_[_tid_] = temp_stencil_117;
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


__global__ void kernel_111(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_114, bool _odd_)
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
            _temp_result_ = _kernel_result_114[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_6_(_env_, _kernel_result_114[_tid_], _kernel_result_114[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_114[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_6_(_env_, _kernel_result_114[_tid_], _kernel_result_114[_tid_ + _num_threads_]);
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




__global__ void kernel_123(environment_t *_env_, int _num_threads_, int *_result_, int *_array_125_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_125_[_tid_];
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


__global__ void kernel_121(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_124)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_126;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_126 = _block_k_11_(_env_, _kernel_result_124[(temp_stencil_dim_0 + -1) * 1], _kernel_result_124[(temp_stencil_dim_0 + 0) * 1], _kernel_result_124[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_126 = 1;
    }
        
        _result_[_tid_] = temp_stencil_126;
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


__global__ void kernel_119(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_122, bool _odd_)
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
            _temp_result_ = _kernel_result_122[_tid_];
        }
        else if (num_args == 2)
        {
            _temp_result_ = _block_k_14_(_env_, _kernel_result_122[_tid_], _kernel_result_122[_tid_ + _num_threads_]);
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
                sdata[thread_idx] = _kernel_result_122[_tid_];
            }
            else
            {
                sdata[thread_idx] = _block_k_14_(_env_, _kernel_result_122[_tid_], _kernel_result_122[_tid_ + _num_threads_]);
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


__global__ void kernel_128(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_132(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_130(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_133)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_134;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_134 = _block_k_4_(_env_, _kernel_result_133[(temp_stencil_dim_0 + -1) * 1], _kernel_result_133[(temp_stencil_dim_0 + 0) * 1], _kernel_result_133[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_134 = 1;
    }
        
        _result_[_tid_] = temp_stencil_134;
    }
}




__global__ void kernel_137(environment_t *_env_, int _num_threads_, int *_result_, int *_array_139_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_139_[_tid_];
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


__global__ void kernel_135(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_138)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_140;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_140 = _block_k_11_(_env_, _kernel_result_138[(temp_stencil_dim_0 + -1) * 1], _kernel_result_138[(temp_stencil_dim_0 + 0) * 1], _kernel_result_138[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_140 = 1;
    }
        
        _result_[_tid_] = temp_stencil_140;
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


__global__ void kernel_141(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_145(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_143(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_146)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_147;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_147 = _block_k_4_(_env_, _kernel_result_146[(temp_stencil_dim_0 + -1) * 1], _kernel_result_146[(temp_stencil_dim_0 + 0) * 1], _kernel_result_146[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_147 = 1;
    }
        
        _result_[_tid_] = temp_stencil_147;
    }
}




__global__ void kernel_150(environment_t *_env_, int _num_threads_, int *_result_, int *_array_152_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_152_[_tid_];
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


__global__ void kernel_148(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_151)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_153;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_153 = _block_k_11_(_env_, _kernel_result_151[(temp_stencil_dim_0 + -1) * 1], _kernel_result_151[(temp_stencil_dim_0 + 0) * 1], _kernel_result_151[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_153 = 1;
    }
        
        _result_[_tid_] = temp_stencil_153;
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


__global__ void kernel_154(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_158(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_156(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_159)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_160;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_160 = _block_k_4_(_env_, _kernel_result_159[(temp_stencil_dim_0 + -1) * 1], _kernel_result_159[(temp_stencil_dim_0 + 0) * 1], _kernel_result_159[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_160 = 1;
    }
        
        _result_[_tid_] = temp_stencil_160;
    }
}




__global__ void kernel_163(environment_t *_env_, int _num_threads_, int *_result_, int *_array_165_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_165_[_tid_];
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


__global__ void kernel_161(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_164)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_166;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_166 = _block_k_11_(_env_, _kernel_result_164[(temp_stencil_dim_0 + -1) * 1], _kernel_result_164[(temp_stencil_dim_0 + 0) * 1], _kernel_result_164[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_166 = 1;
    }
        
        _result_[_tid_] = temp_stencil_166;
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


__global__ void kernel_167(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_171(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_169(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_172)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_173;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_173 = _block_k_4_(_env_, _kernel_result_172[(temp_stencil_dim_0 + -1) * 1], _kernel_result_172[(temp_stencil_dim_0 + 0) * 1], _kernel_result_172[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_173 = 1;
    }
        
        _result_[_tid_] = temp_stencil_173;
    }
}




__global__ void kernel_176(environment_t *_env_, int _num_threads_, int *_result_, int *_array_178_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_178_[_tid_];
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


__global__ void kernel_174(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_177)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_179;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_179 = _block_k_11_(_env_, _kernel_result_177[(temp_stencil_dim_0 + -1) * 1], _kernel_result_177[(temp_stencil_dim_0 + 0) * 1], _kernel_result_177[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_179 = 1;
    }
        
        _result_[_tid_] = temp_stencil_179;
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


__global__ void kernel_180(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_184(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_182(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_185)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_186;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_186 = _block_k_4_(_env_, _kernel_result_185[(temp_stencil_dim_0 + -1) * 1], _kernel_result_185[(temp_stencil_dim_0 + 0) * 1], _kernel_result_185[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_186 = 1;
    }
        
        _result_[_tid_] = temp_stencil_186;
    }
}




__global__ void kernel_189(environment_t *_env_, int _num_threads_, int *_result_, int *_array_191_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_191_[_tid_];
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


__global__ void kernel_187(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_190)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_192;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 1;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 90210)
    {
        // All value indices within bounds
        
        temp_stencil_192 = _block_k_11_(_env_, _kernel_result_190[(temp_stencil_dim_0 + -1) * 1], _kernel_result_190[(temp_stencil_dim_0 + 0) * 1], _kernel_result_190[(temp_stencil_dim_0 + 1) * 1]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_192 = 1;
    }
        
        _result_[_tid_] = temp_stencil_192;
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
                            case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_13 = union_t(18, union_v_t::from_pointer((void *) new array_command_14(NULL, (array_command_11 *) _polytemp_expr_14.value.pointer))); break;
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
                                timeStartMeasure();
                        int * _kernel_result_43;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_43, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_43);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_42<<<353, 256>>>(dev_env, 90210, _kernel_result_43);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_44;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_44, (sizeof(int) * 45105)));
                        program_result->device_allocations->push_back(_kernel_result_44);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_40<<<177, 256>>>(dev_env, 45105, _kernel_result_44, _kernel_result_43, false);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        kernel_40<<<1, 89>>>(dev_env, 89, _kernel_result_44, _kernel_result_44, true);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_44;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_43 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_43));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_43),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                        timeStartMeasure();
                    
                        if (_kernel_result_44 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_44));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_44),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                    case 16: /* [Ikra::Symbolic::ArrayReduceCommand, size = 1] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayReduceCommand, size = 1]: [SendNode: [LVarReadNode: _ssa_var_a_1].preduce()]
                    
                        array_command_6 * cmd = (array_command_6 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_50;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_50, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_50);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_49<<<353, 256>>>(dev_env, 90210, _kernel_result_50);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_48;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_48, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_48);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_47<<<353, 256>>>(dev_env, 90210, _kernel_result_48, _kernel_result_50);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_52;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_52, (sizeof(int) * 45105)));
                        program_result->device_allocations->push_back(_kernel_result_52);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_45<<<177, 256>>>(dev_env, 45105, _kernel_result_52, _kernel_result_48, false);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        kernel_45<<<1, 89>>>(dev_env, 89, _kernel_result_52, _kernel_result_52, true);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_52;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_50 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_50));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_50),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                        timeStartMeasure();
                    
                        if (_kernel_result_48 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_48));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_48),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                        timeStartMeasure();
                    
                        if (_kernel_result_52 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_52));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_52),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                    case 18: /* [Ikra::Symbolic::ArrayReduceCommand, size = 1] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayReduceCommand, size = 1]: [SendNode: [LVarReadNode: _ssa_var_a_1].preduce()]
                    
                        array_command_14 * cmd = (array_command_14 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_58;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_58, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_58);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_57<<<353, 256>>>(dev_env, 90210, _kernel_result_58, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_56;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_56, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_56);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_55<<<353, 256>>>(dev_env, 90210, _kernel_result_56, _kernel_result_58);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_61;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_61, (sizeof(int) * 45105)));
                        program_result->device_allocations->push_back(_kernel_result_61);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_53<<<177, 256>>>(dev_env, 45105, _kernel_result_61, _kernel_result_56, false);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        kernel_53<<<1, 89>>>(dev_env, 89, _kernel_result_61, _kernel_result_61, true);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_61;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_58 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_58));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_58),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                        timeStartMeasure();
                    
                        if (_kernel_result_56 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_56));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_56),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                        timeStartMeasure();
                    
                        if (_kernel_result_61 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_61));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_61),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 1);
                    }); break;
                }
            }
            _polytemp_result_11;
        });
            int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);
        
            timeStartMeasure();
            checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
            timeReportMeasure(program_result, transfer_memory);
        
            variable_size_array_t((void *) tmp_result, device_array.size);
        }).content)[0] < 10000000)))
        {
            _ssa_var_a_2 = union_t(17, union_v_t::from_pointer((void *) new array_command_11(NULL, new array_command_10(NULL, ({
                variable_size_array_t _polytemp_result_27;
                {
                    union_t _polytemp_expr_28 = _ssa_var_a_1;
                    switch (_polytemp_expr_28.class_id)
                    {
                        case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 90210]
                        
                            array_command_2 * cmd = (array_command_2 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_129;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_129, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_129);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_128<<<353, 256>>>(dev_env, 90210, _kernel_result_129);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_129;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 90210);
                        }); break;
                        case 15: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_133;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_133, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_133);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_132<<<353, 256>>>(dev_env, 90210, _kernel_result_133);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            int * _kernel_result_131;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_131, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_131);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_130<<<353, 256>>>(dev_env, 90210, _kernel_result_131, _kernel_result_133);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_131;
                        
                                    timeStartMeasure();
                        
                            if (_kernel_result_133 != cmd->result) {
                                // Don't free memory if it is the result. There is already a similar check in
                                // program_builder (free all except for last). However, this check is not sufficient in
                                // case the same array is reused!
                        
                                checkErrorReturn(program_result, cudaFree(_kernel_result_133));
                                // Remove from list of allocations
                                program_result->device_allocations->erase(
                                    std::remove(
                                        program_result->device_allocations->begin(),
                                        program_result->device_allocations->end(),
                                        _kernel_result_133),
                                    program_result->device_allocations->end());
                            }
                        
                            timeReportMeasure(program_result, free_memory);
                        
                            }
                        
                            variable_size_array_t((void *) cmd->result, 90210);
                        }); break;
                        case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                        
                            array_command_11 * cmd = (array_command_11 *) _polytemp_expr_28.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_138;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_138, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_138);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_137<<<353, 256>>>(dev_env, 90210, _kernel_result_138, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            int * _kernel_result_136;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_136, (sizeof(int) * 90210)));
                            program_result->device_allocations->push_back(_kernel_result_136);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_135<<<353, 256>>>(dev_env, 90210, _kernel_result_136, _kernel_result_138);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_136;
                        
                                    timeStartMeasure();
                        
                            if (_kernel_result_138 != cmd->result) {
                                // Don't free memory if it is the result. There is already a similar check in
                                // program_builder (free all except for last). However, this check is not sufficient in
                                // case the same array is reused!
                        
                                checkErrorReturn(program_result, cudaFree(_kernel_result_138));
                                // Remove from list of allocations
                                program_result->device_allocations->erase(
                                    std::remove(
                                        program_result->device_allocations->begin(),
                                        program_result->device_allocations->end(),
                                        _kernel_result_138),
                                    program_result->device_allocations->end());
                            }
                        
                            timeReportMeasure(program_result, free_memory);
                        
                            }
                        
                            variable_size_array_t((void *) cmd->result, 90210);
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
                                timeStartMeasure();
                        int * _kernel_result_181;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_181, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_181);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_180<<<353, 256>>>(dev_env, 90210, _kernel_result_181);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_181;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 90210);
                    }); break;
                    case 15: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_35 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                    
                        array_command_4 * cmd = (array_command_4 *) _polytemp_expr_36.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_185;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_185, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_185);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_184<<<353, 256>>>(dev_env, 90210, _kernel_result_185);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_183;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_183, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_183);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_182<<<353, 256>>>(dev_env, 90210, _kernel_result_183, _kernel_result_185);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_183;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_185 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_185));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_185),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 90210);
                    }); break;
                    case 17: /* [Ikra::Symbolic::ArrayStencilCommand, size = 90210] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_35 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 90210]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pstencil([ArrayNode: [<-1>, <0>, <1>]]; <1>)]
                    
                        array_command_11 * cmd = (array_command_11 *) _polytemp_expr_36.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_190;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_190, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_190);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_189<<<353, 256>>>(dev_env, 90210, _kernel_result_190, ((int *) ((int *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_188;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_188, (sizeof(int) * 90210)));
                        program_result->device_allocations->push_back(_kernel_result_188);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_187<<<353, 256>>>(dev_env, 90210, _kernel_result_188, _kernel_result_190);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_188;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_190 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_190));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_190),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 90210);
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
