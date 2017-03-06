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
struct indexed_struct_3_lt_int_int_int_gt_t
{
    int field_0;
int field_1;
int field_2;
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
    indexed_struct_3_lt_int_int_int_gt_t *result;
    __host__ __device__ array_command_1(indexed_struct_3_lt_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_2 {
    // Ikra::Symbolic::ArrayCombineCommand
    float *result;
    array_command_1 *input_0;
    __host__ __device__ array_command_2(float *result = NULL, array_command_1 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_3 {
    // Ikra::Symbolic::ArrayStencilCommand
    float *result;
    array_command_2 *input_0;
    __host__ __device__ array_command_3(float *result = NULL, array_command_2 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_5 {
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    float *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_5(float *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayStencilCommand
    float *result;
    array_command_5 *input_0;
    __host__ __device__ array_command_6(float *result = NULL, array_command_5 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct environment_struct
{
    int l2_k_max;
};

// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_5(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ float _block_k_3_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_3(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_6)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_7;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_7 = _block_k_3_(_env_, _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_6[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_6[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_7 = 0;
    }
        
        _result_[_tid_] = temp_stencil_7;
    }
}




__global__ void kernel_10(environment_t *_env_, int _num_threads_, float *_result_, float *_array_12_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_12_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ float _block_k_6_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_8(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_11)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_13;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_13 = _block_k_6_(_env_, _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_13 = 0;
    }
        
        _result_[_tid_] = temp_stencil_13;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_14(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_18(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ float _block_k_3_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_16(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_19)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_20;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_20 = _block_k_3_(_env_, _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_19[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_19[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_20 = 0;
    }
        
        _result_[_tid_] = temp_stencil_20;
    }
}




__global__ void kernel_23(environment_t *_env_, int _num_threads_, float *_result_, float *_array_25_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_25_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ float _block_k_6_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_21(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_24)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_26;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_26 = _block_k_6_(_env_, _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_24[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_24[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_26 = 0;
    }
        
        _result_[_tid_] = temp_stencil_26;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_27(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_31(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ float _block_k_3_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_29(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_32)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_33;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_33 = _block_k_3_(_env_, _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_32[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_32[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_33 = 0;
    }
        
        _result_[_tid_] = temp_stencil_33;
    }
}




__global__ void kernel_36(environment_t *_env_, int _num_threads_, float *_result_, float *_array_38_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_38_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ float _block_k_6_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_34(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_37)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_39;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_39 = _block_k_6_(_env_, _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_37[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_37[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_39 = 0;
    }
        
        _result_[_tid_] = temp_stencil_39;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_40(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_44(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ float _block_k_3_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_42(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_45)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_46;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_46 = _block_k_3_(_env_, _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_45[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_45[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_46 = 0;
    }
        
        _result_[_tid_] = temp_stencil_46;
    }
}




__global__ void kernel_49(environment_t *_env_, int _num_threads_, float *_result_, float *_array_51_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_51_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ float _block_k_6_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_47(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_50)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_52;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_52 = _block_k_6_(_env_, _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_50[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_50[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_52 = 0;
    }
        
        _result_[_tid_] = temp_stencil_52;
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_53(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    int lex_k_max = _env_->l2_k_max;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((lex_k_max - 1)))) / ((lex_k_max - 1)));
        }
    }
}

#endif


__global__ void kernel_57(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ float _block_k_3_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_55(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_58)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_59;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_59 = _block_k_3_(_env_, _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_58[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_58[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_59 = 0;
    }
        
        _result_[_tid_] = temp_stencil_59;
    }
}




__global__ void kernel_62(environment_t *_env_, int _num_threads_, float *_result_, float *_array_64_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_64_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ float _block_k_6_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    {
        return (values[0] + ((((0.8 * ((((((((((((((((((((((((0.0 * values[1])) + ((0.0 * values[2])))) + ((0.0 * values[3])))) + ((0.0 * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((0.0 * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((0.0 * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((1.0 * values[16])))) + ((1.0 * values[17])))) + ((1.0 * values[18])))) + 0.0)) * ((1.0 / 6.0)))) - values[0])))) * 1.0)));
    }
}

#endif


__global__ void kernel_60(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_63)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_65;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_65 = _block_k_6_(_env_, _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_63[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_63[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225]);
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_65 = 0;
    }
        
        _result_[_tid_] = temp_stencil_65;
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
    array_command_2 * p = new array_command_2();
    int i;
    union_t _ssa_var_next_p_2;
    union_t _ssa_var_next_p_1;
    {
        _ssa_var_next_p_1 = union_t(10, union_v_t::from_pointer((void *) p));
        for (i = 0; i <= (200 - 1); i++)
        {
            _ssa_var_next_p_2 = union_t(12, union_v_t::from_pointer((void *) new array_command_6(NULL, new array_command_5(NULL, ({
                variable_size_array_t _polytemp_result_1;
                {
                    union_t _polytemp_expr_2 = _ssa_var_next_p_1;
                    switch (_polytemp_expr_2.class_id)
                    {
                        case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 545025]
                        
                            array_command_2 * cmd = (array_command_2 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            float * _kernel_result_2;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_2);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_1<<<2130, 256>>>(dev_env, 545025, _kernel_result_2);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_2;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 545025);
                        }); break;
                        case 11: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>)]
                        
                            array_command_3 * cmd = (array_command_3 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            float * _kernel_result_6;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_6);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_5<<<2130, 256>>>(dev_env, 545025, _kernel_result_6);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            float * _kernel_result_4;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_4, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_4);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_3<<<2130, 256>>>(dev_env, 545025, _kernel_result_4, _kernel_result_6);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_4;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 545025);
                        }); break;
                        case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>)]
                        
                            array_command_6 * cmd = (array_command_6 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            float * _kernel_result_11;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_11, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_11);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_10<<<2130, 256>>>(dev_env, 545025, _kernel_result_11, ((float *) cmd->input_0->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            float * _kernel_result_9;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_9, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_9);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_8<<<2130, 256>>>(dev_env, 545025, _kernel_result_9, _kernel_result_11);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_9;
                            }
                        
                            variable_size_array_t((void *) cmd->result, 545025);
                        }); break;
                    }
                }
                _polytemp_result_1;
            })))));
            _ssa_var_next_p_1 = _ssa_var_next_p_2;
        }
        i--;
        return ({
            variable_size_array_t _polytemp_result_9;
            {
                union_t _polytemp_expr_10 = _ssa_var_next_p_1;
                switch (_polytemp_expr_10.class_id)
                {
                    case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 545025]
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        float * _kernel_result_54;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_54, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_54);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_53<<<2130, 256>>>(dev_env, 545025, _kernel_result_54);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_54;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 545025);
                    }); break;
                    case 11: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>)]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        float * _kernel_result_58;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_58, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_58);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_57<<<2130, 256>>>(dev_env, 545025, _kernel_result_58);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        float * _kernel_result_56;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_56, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_56);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_55<<<2130, 256>>>(dev_env, 545025, _kernel_result_56, _kernel_result_58);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_56;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 545025);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>)]
                    
                        array_command_6 * cmd = (array_command_6 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        float * _kernel_result_63;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_63, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_63);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_62<<<2130, 256>>>(dev_env, 545025, _kernel_result_63, ((float *) cmd->input_0->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        float * _kernel_result_61;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_61, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_61);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_60<<<2130, 256>>>(dev_env, 545025, _kernel_result_61, _kernel_result_63);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_61;
                        }
                    
                        variable_size_array_t((void *) cmd->result, 545025);
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
    float * tmp_result = (float *) malloc(sizeof(float) * device_array.size);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(float) * device_array.size, cudaMemcpyDeviceToHost));
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
