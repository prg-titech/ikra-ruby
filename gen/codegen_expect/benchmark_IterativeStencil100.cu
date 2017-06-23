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
struct array_command_3 {
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    int *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_3(int *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_4_lt_int_int_int_int_gt_t *result;
    __host__ __device__ array_command_5(indexed_struct_4_lt_int_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayStencilCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_4(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
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
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_1 = ((indices.field_1 % 4));
        (_temp_var_1 == 0 ? indices.field_0 : (_temp_var_1 == 1 ? indices.field_1 : (_temp_var_1 == 2 ? indices.field_2 : (_temp_var_1 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    {
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_2 = ((indices.field_1 % 4));
        (_temp_var_2 == 0 ? indices.field_0 : (_temp_var_2 == 1 ? indices.field_1 : (_temp_var_2 == 2 ? indices.field_2 : (_temp_var_2 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_3(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
}




__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_, int *_array_7_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_7_[_tid_];
    }
}




__global__ void kernel_10(environment_t *_env_, int _num_threads_, int *_result_, int *_array_12_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_12_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_5 = ((({ int _temp_var_6 = ((({ int _temp_var_7 = ((values[2] % 4));
        (_temp_var_7 == 0 ? indices.field_0 : (_temp_var_7 == 1 ? indices.field_1 : (_temp_var_7 == 2 ? indices.field_2 : (_temp_var_7 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_6 == 0 ? indices.field_0 : (_temp_var_6 == 1 ? indices.field_1 : (_temp_var_6 == 2 ? indices.field_2 : (_temp_var_6 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_5 == 0 ? indices.field_0 : (_temp_var_5 == 1 ? indices.field_1 : (_temp_var_5 == 2 ? indices.field_2 : (_temp_var_5 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_8(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_11)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_13;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 3000000;
int temp_stencil_dim_1 = (_tid_ / 6000) % 500;
int temp_stencil_dim_2 = (_tid_ / 12) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 12;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 12)
    {
        // All value indices within bounds
        
        temp_stencil_13 = _block_k_4_(_env_, _kernel_result_11[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], _kernel_result_11[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 0) * 3000000], _kernel_result_11[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 1) * 3000000], _kernel_result_11[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + -1) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_13 = 37;
    }
        
        _result_[_tid_] = temp_stencil_13;
    }
}




__global__ void kernel_14(environment_t *_env_, int _num_threads_, int *_result_, int *_array_16_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_16_[_tid_];
    }
}




__global__ void kernel_19(environment_t *_env_, int _num_threads_, int *_result_, int *_array_21_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_21_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_10 = ((({ int _temp_var_11 = ((({ int _temp_var_12 = ((values[2] % 4));
        (_temp_var_12 == 0 ? indices.field_0 : (_temp_var_12 == 1 ? indices.field_1 : (_temp_var_12 == 2 ? indices.field_2 : (_temp_var_12 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_11 == 0 ? indices.field_0 : (_temp_var_11 == 1 ? indices.field_1 : (_temp_var_11 == 2 ? indices.field_2 : (_temp_var_11 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_10 == 0 ? indices.field_0 : (_temp_var_10 == 1 ? indices.field_1 : (_temp_var_10 == 2 ? indices.field_2 : (_temp_var_10 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_17(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_20)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_22;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 3000000;
int temp_stencil_dim_1 = (_tid_ / 6000) % 500;
int temp_stencil_dim_2 = (_tid_ / 12) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 12;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 12)
    {
        // All value indices within bounds
        
        temp_stencil_22 = _block_k_4_(_env_, _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 0) * 3000000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 1) * 3000000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + -1) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_22 = 37;
    }
        
        _result_[_tid_] = temp_stencil_22;
    }
}




__global__ void kernel_23(environment_t *_env_, int _num_threads_, int *_result_, int *_array_25_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_25_[_tid_];
    }
}




__global__ void kernel_28(environment_t *_env_, int _num_threads_, int *_result_, int *_array_30_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_30_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_15 = ((({ int _temp_var_16 = ((({ int _temp_var_17 = ((values[2] % 4));
        (_temp_var_17 == 0 ? indices.field_0 : (_temp_var_17 == 1 ? indices.field_1 : (_temp_var_17 == 2 ? indices.field_2 : (_temp_var_17 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_16 == 0 ? indices.field_0 : (_temp_var_16 == 1 ? indices.field_1 : (_temp_var_16 == 2 ? indices.field_2 : (_temp_var_16 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_15 == 0 ? indices.field_0 : (_temp_var_15 == 1 ? indices.field_1 : (_temp_var_15 == 2 ? indices.field_2 : (_temp_var_15 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_26(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_29)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_31;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 3000000;
int temp_stencil_dim_1 = (_tid_ / 6000) % 500;
int temp_stencil_dim_2 = (_tid_ / 12) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 12;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 12)
    {
        // All value indices within bounds
        
        temp_stencil_31 = _block_k_4_(_env_, _kernel_result_29[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], _kernel_result_29[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 0) * 3000000], _kernel_result_29[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 1) * 3000000], _kernel_result_29[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + -1) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_31 = 37;
    }
        
        _result_[_tid_] = temp_stencil_31;
    }
}




__global__ void kernel_32(environment_t *_env_, int _num_threads_, int *_result_, int *_array_34_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_34_[_tid_];
    }
}




__global__ void kernel_37(environment_t *_env_, int _num_threads_, int *_result_, int *_array_39_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_39_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_20 = ((({ int _temp_var_21 = ((({ int _temp_var_22 = ((values[2] % 4));
        (_temp_var_22 == 0 ? indices.field_0 : (_temp_var_22 == 1 ? indices.field_1 : (_temp_var_22 == 2 ? indices.field_2 : (_temp_var_22 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_21 == 0 ? indices.field_0 : (_temp_var_21 == 1 ? indices.field_1 : (_temp_var_21 == 2 ? indices.field_2 : (_temp_var_21 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_20 == 0 ? indices.field_0 : (_temp_var_20 == 1 ? indices.field_1 : (_temp_var_20 == 2 ? indices.field_2 : (_temp_var_20 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_35(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_38)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_40;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 3000000;
int temp_stencil_dim_1 = (_tid_ / 6000) % 500;
int temp_stencil_dim_2 = (_tid_ / 12) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 12;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 12)
    {
        // All value indices within bounds
        
        temp_stencil_40 = _block_k_4_(_env_, _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 0) * 3000000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 1) * 3000000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + -1) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_40 = 37;
    }
        
        _result_[_tid_] = temp_stencil_40;
    }
}




__global__ void kernel_41(environment_t *_env_, int _num_threads_, int *_result_, int *_array_43_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_43_[_tid_];
    }
}




__global__ void kernel_46(environment_t *_env_, int _num_threads_, int *_result_, int *_array_48_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_48_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_27 = ((({ int _temp_var_28 = ((({ int _temp_var_29 = ((values[2] % 4));
        (_temp_var_29 == 0 ? indices.field_0 : (_temp_var_29 == 1 ? indices.field_1 : (_temp_var_29 == 2 ? indices.field_2 : (_temp_var_29 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_28 == 0 ? indices.field_0 : (_temp_var_28 == 1 ? indices.field_1 : (_temp_var_28 == 2 ? indices.field_2 : (_temp_var_28 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_27 == 0 ? indices.field_0 : (_temp_var_27 == 1 ? indices.field_1 : (_temp_var_27 == 2 ? indices.field_2 : (_temp_var_27 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_44(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_47)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_49;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 3000000;
int temp_stencil_dim_1 = (_tid_ / 6000) % 500;
int temp_stencil_dim_2 = (_tid_ / 12) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 12;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 12)
    {
        // All value indices within bounds
        
        temp_stencil_49 = _block_k_4_(_env_, _kernel_result_47[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], _kernel_result_47[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 0) * 3000000], _kernel_result_47[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + 0) * 6000 + (temp_stencil_dim_0 + 1) * 3000000], _kernel_result_47[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 12 + (temp_stencil_dim_1 + -1) * 6000 + (temp_stencil_dim_0 + -1) * 3000000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_49 = 37;
    }
        
        _result_[_tid_] = temp_stencil_49;
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
    array_command_2 * x = new array_command_2();
    union_t _ssa_var_old_old_data_3;
    array_command_4 * _ssa_var_y_6;
    union_t _ssa_var_old_data_5;
    union_t _ssa_var_old_old_data_4;
    int r;
    union_t _ssa_var_old_data_2;
    union_t _ssa_var_y_1;
    {
        _ssa_var_y_1 = union_t(10, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 60000000]
        
            array_command_2 * cmd = x;
        
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
        }))));
        _ssa_var_old_data_2 = union_t(11, union_v_t::from_pointer((void *) x));
        _ssa_var_old_old_data_3 = union_t(11, union_v_t::from_pointer((void *) x));
        for (r = 0; r <= (200 - 1); r++)
        {
            _ssa_var_old_old_data_4 = _ssa_var_old_data_2;
            _ssa_var_old_data_5 = _ssa_var_y_1;
            _ssa_var_y_6 = new array_command_4(NULL, new array_command_3(NULL, ({
                variable_size_array_t _polytemp_result_3;
                {
                    union_t _polytemp_expr_4 = _ssa_var_y_1;
                    switch (_polytemp_expr_4.class_id)
                    {
                        case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_3 = ({
                            // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 60000000]
                        
                            array_command_3 * cmd = (array_command_3 *) _polytemp_expr_4.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_6;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 60000000)));
                            program_result->device_allocations->push_back(_kernel_result_6);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_5<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_6, ((int *) cmd->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_6;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 60000000);
                        }); break;
                        case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_3 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 60000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_4.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_11;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_11, (sizeof(int) * 60000000)));
                            program_result->device_allocations->push_back(_kernel_result_11);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_10<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_11, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            int * _kernel_result_9;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_9, (sizeof(int) * 60000000)));
                            program_result->device_allocations->push_back(_kernel_result_9);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_8<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_9, _kernel_result_11);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            checkErrorReturn(program_result, cudaFree(_kernel_result_11));
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_11),
                                program_result->device_allocations->end());
                            timeReportMeasure(program_result, free_memory);
                        
                                cmd->result = _kernel_result_9;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 60000000);
                        }); break;
                    }
                }
                _polytemp_result_3;
            })));
            if (((r > 1)))
            {
                ({
                    bool _polytemp_result_23;
                    {
                        union_t _polytemp_expr_24 = _ssa_var_old_old_data_4;
                        switch (_polytemp_expr_24.class_id)
                        {
                            case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_23 = ({
                                array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_24.value.pointer;
                            
                                timeStartMeasure();
                                bool freed_memory = false;
                            
                                if (cmd_to_free->result != 0) {
                                    checkErrorReturn(program_result, cudaFree(cmd_to_free->result));;
                            
                                    // Remove from list of allocations
                                    program_result->device_allocations->erase(
                                        std::remove(
                                            program_result->device_allocations->begin(),
                                            program_result->device_allocations->end(),
                                            cmd_to_free->result),
                                        program_result->device_allocations->end());
                            
                                    freed_memory = true;
                                }
                            
                                timeReportMeasure(program_result, free_memory);
                                
                                freed_memory;
                            }); break;
                            case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_23 = ({
                                array_command_3 * cmd_to_free = (array_command_3 *) _polytemp_expr_24.value.pointer;
                            
                                timeStartMeasure();
                                bool freed_memory = false;
                            
                                if (cmd_to_free->result != 0) {
                                    checkErrorReturn(program_result, cudaFree(cmd_to_free->result));;
                            
                                    // Remove from list of allocations
                                    program_result->device_allocations->erase(
                                        std::remove(
                                            program_result->device_allocations->begin(),
                                            program_result->device_allocations->end(),
                                            cmd_to_free->result),
                                        program_result->device_allocations->end());
                            
                                    freed_memory = true;
                                }
                            
                                timeReportMeasure(program_result, free_memory);
                                
                                freed_memory;
                            }); break;
                            case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_23 = ({
                                array_command_4 * cmd_to_free = (array_command_4 *) _polytemp_expr_24.value.pointer;
                            
                                timeStartMeasure();
                                bool freed_memory = false;
                            
                                if (cmd_to_free->result != 0) {
                                    checkErrorReturn(program_result, cudaFree(cmd_to_free->result));;
                            
                                    // Remove from list of allocations
                                    program_result->device_allocations->erase(
                                        std::remove(
                                            program_result->device_allocations->begin(),
                                            program_result->device_allocations->end(),
                                            cmd_to_free->result),
                                        program_result->device_allocations->end());
                            
                                    freed_memory = true;
                                }
                            
                                timeReportMeasure(program_result, free_memory);
                                
                                freed_memory;
                            }); break;
                        }
                    }
                    _polytemp_result_23;
                });
            }
            else
            {
            
            }
            _ssa_var_y_1 = union_t(12, union_v_t::from_pointer((void *) _ssa_var_y_6));
            _ssa_var_old_data_2 = _ssa_var_old_data_5;
            _ssa_var_old_old_data_3 = _ssa_var_old_old_data_4;
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_25;
            {
                union_t _polytemp_expr_26 = _ssa_var_y_1;
                switch (_polytemp_expr_26.class_id)
                {
                    case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_25 = ({
                        // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 60000000]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_26.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_42;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_42, (sizeof(int) * 60000000)));
                        program_result->device_allocations->push_back(_kernel_result_42);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_41<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_42, ((int *) cmd->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_42;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 60000000);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 60000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_25 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 60000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_4 * cmd = (array_command_4 *) _polytemp_expr_26.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_47;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_47, (sizeof(int) * 60000000)));
                        program_result->device_allocations->push_back(_kernel_result_47);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_46<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_47, ((int *) ((int *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_45;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_45, (sizeof(int) * 60000000)));
                        program_result->device_allocations->push_back(_kernel_result_45);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_44<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_45, _kernel_result_47);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        checkErrorReturn(program_result, cudaFree(_kernel_result_47));
                        program_result->device_allocations->erase(
                            std::remove(
                                program_result->device_allocations->begin(),
                                program_result->device_allocations->end(),
                                _kernel_result_47),
                            program_result->device_allocations->end());
                        timeReportMeasure(program_result, free_memory);
                    
                            cmd->result = _kernel_result_45;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 60000000);
                    }); break;
                }
            }
            _polytemp_result_25;
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
