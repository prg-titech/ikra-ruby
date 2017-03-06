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


__global__ void kernel_401(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_2 = ((({ int _temp_var_3 = ((({ int _temp_var_4 = ((values[2] % 4));
        (_temp_var_4 == 0 ? indices.field_0 : (_temp_var_4 == 1 ? indices.field_1 : (_temp_var_4 == 2 ? indices.field_2 : (_temp_var_4 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_3 == 0 ? indices.field_0 : (_temp_var_3 == 1 ? indices.field_1 : (_temp_var_3 == 2 ? indices.field_2 : (_temp_var_3 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_2 == 0 ? indices.field_0 : (_temp_var_2 == 1 ? indices.field_1 : (_temp_var_2 == 2 ? indices.field_2 : (_temp_var_2 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_399(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_402)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_403;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_403 = _block_k_3_(_env_, _kernel_result_402[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_402[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_402[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_402[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_403 = 37;
    }
        
        _result_[_tid_] = temp_stencil_403;
    }
}



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
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


__global__ void kernel_397(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_400)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_404;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_404 = _block_k_5_(_env_, _kernel_result_400[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_400[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_400[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_400[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_404 = 37;
    }
        
        _result_[_tid_] = temp_stencil_404;
    }
}



// TODO: There should be a better to check if _block_k_7_ is already defined
#ifndef _block_k_7__func
#define _block_k_7__func
__device__ int _block_k_7_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_8 = ((({ int _temp_var_9 = ((({ int _temp_var_10 = ((values[2] % 4));
        (_temp_var_10 == 0 ? indices.field_0 : (_temp_var_10 == 1 ? indices.field_1 : (_temp_var_10 == 2 ? indices.field_2 : (_temp_var_10 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_9 == 0 ? indices.field_0 : (_temp_var_9 == 1 ? indices.field_1 : (_temp_var_9 == 2 ? indices.field_2 : (_temp_var_9 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_8 == 0 ? indices.field_0 : (_temp_var_8 == 1 ? indices.field_1 : (_temp_var_8 == 2 ? indices.field_2 : (_temp_var_8 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_395(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_398)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_405;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_405 = _block_k_7_(_env_, _kernel_result_398[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_398[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_398[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_398[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_405 = 37;
    }
        
        _result_[_tid_] = temp_stencil_405;
    }
}



// TODO: There should be a better to check if _block_k_9_ is already defined
#ifndef _block_k_9__func
#define _block_k_9__func
__device__ int _block_k_9_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_11 = ((({ int _temp_var_12 = ((({ int _temp_var_13 = ((values[2] % 4));
        (_temp_var_13 == 0 ? indices.field_0 : (_temp_var_13 == 1 ? indices.field_1 : (_temp_var_13 == 2 ? indices.field_2 : (_temp_var_13 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_12 == 0 ? indices.field_0 : (_temp_var_12 == 1 ? indices.field_1 : (_temp_var_12 == 2 ? indices.field_2 : (_temp_var_12 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_11 == 0 ? indices.field_0 : (_temp_var_11 == 1 ? indices.field_1 : (_temp_var_11 == 2 ? indices.field_2 : (_temp_var_11 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_393(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_396)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_406;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_406 = _block_k_9_(_env_, _kernel_result_396[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_396[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_396[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_396[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_406 = 37;
    }
        
        _result_[_tid_] = temp_stencil_406;
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_14 = ((({ int _temp_var_15 = ((({ int _temp_var_16 = ((values[2] % 4));
        (_temp_var_16 == 0 ? indices.field_0 : (_temp_var_16 == 1 ? indices.field_1 : (_temp_var_16 == 2 ? indices.field_2 : (_temp_var_16 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_15 == 0 ? indices.field_0 : (_temp_var_15 == 1 ? indices.field_1 : (_temp_var_15 == 2 ? indices.field_2 : (_temp_var_15 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_14 == 0 ? indices.field_0 : (_temp_var_14 == 1 ? indices.field_1 : (_temp_var_14 == 2 ? indices.field_2 : (_temp_var_14 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_391(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_394)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_407;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_407 = _block_k_11_(_env_, _kernel_result_394[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_394[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_394[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_394[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_407 = 37;
    }
        
        _result_[_tid_] = temp_stencil_407;
    }
}



// TODO: There should be a better to check if _block_k_13_ is already defined
#ifndef _block_k_13__func
#define _block_k_13__func
__device__ int _block_k_13_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_17 = ((({ int _temp_var_18 = ((({ int _temp_var_19 = ((values[2] % 4));
        (_temp_var_19 == 0 ? indices.field_0 : (_temp_var_19 == 1 ? indices.field_1 : (_temp_var_19 == 2 ? indices.field_2 : (_temp_var_19 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_18 == 0 ? indices.field_0 : (_temp_var_18 == 1 ? indices.field_1 : (_temp_var_18 == 2 ? indices.field_2 : (_temp_var_18 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_17 == 0 ? indices.field_0 : (_temp_var_17 == 1 ? indices.field_1 : (_temp_var_17 == 2 ? indices.field_2 : (_temp_var_17 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_389(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_392)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_408;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_408 = _block_k_13_(_env_, _kernel_result_392[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_392[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_392[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_392[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_408 = 37;
    }
        
        _result_[_tid_] = temp_stencil_408;
    }
}



// TODO: There should be a better to check if _block_k_15_ is already defined
#ifndef _block_k_15__func
#define _block_k_15__func
__device__ int _block_k_15_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
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


__global__ void kernel_387(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_390)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_409;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_409 = _block_k_15_(_env_, _kernel_result_390[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_390[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_390[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_390[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_409 = 37;
    }
        
        _result_[_tid_] = temp_stencil_409;
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_23 = ((({ int _temp_var_24 = ((({ int _temp_var_25 = ((values[2] % 4));
        (_temp_var_25 == 0 ? indices.field_0 : (_temp_var_25 == 1 ? indices.field_1 : (_temp_var_25 == 2 ? indices.field_2 : (_temp_var_25 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_24 == 0 ? indices.field_0 : (_temp_var_24 == 1 ? indices.field_1 : (_temp_var_24 == 2 ? indices.field_2 : (_temp_var_24 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_23 == 0 ? indices.field_0 : (_temp_var_23 == 1 ? indices.field_1 : (_temp_var_23 == 2 ? indices.field_2 : (_temp_var_23 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_385(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_388)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_410;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_410 = _block_k_17_(_env_, _kernel_result_388[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_388[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_388[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_388[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_410 = 37;
    }
        
        _result_[_tid_] = temp_stencil_410;
    }
}



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_26 = ((({ int _temp_var_27 = ((({ int _temp_var_28 = ((values[2] % 4));
        (_temp_var_28 == 0 ? indices.field_0 : (_temp_var_28 == 1 ? indices.field_1 : (_temp_var_28 == 2 ? indices.field_2 : (_temp_var_28 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_27 == 0 ? indices.field_0 : (_temp_var_27 == 1 ? indices.field_1 : (_temp_var_27 == 2 ? indices.field_2 : (_temp_var_27 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_26 == 0 ? indices.field_0 : (_temp_var_26 == 1 ? indices.field_1 : (_temp_var_26 == 2 ? indices.field_2 : (_temp_var_26 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_383(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_386)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_411;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_411 = _block_k_19_(_env_, _kernel_result_386[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_386[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_386[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_386[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_411 = 37;
    }
        
        _result_[_tid_] = temp_stencil_411;
    }
}



// TODO: There should be a better to check if _block_k_21_ is already defined
#ifndef _block_k_21__func
#define _block_k_21__func
__device__ int _block_k_21_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_29 = ((({ int _temp_var_30 = ((({ int _temp_var_31 = ((values[2] % 4));
        (_temp_var_31 == 0 ? indices.field_0 : (_temp_var_31 == 1 ? indices.field_1 : (_temp_var_31 == 2 ? indices.field_2 : (_temp_var_31 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_30 == 0 ? indices.field_0 : (_temp_var_30 == 1 ? indices.field_1 : (_temp_var_30 == 2 ? indices.field_2 : (_temp_var_30 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_29 == 0 ? indices.field_0 : (_temp_var_29 == 1 ? indices.field_1 : (_temp_var_29 == 2 ? indices.field_2 : (_temp_var_29 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_381(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_384)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_412;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_412 = _block_k_21_(_env_, _kernel_result_384[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_384[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_384[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_384[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_412 = 37;
    }
        
        _result_[_tid_] = temp_stencil_412;
    }
}



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_32 = ((({ int _temp_var_33 = ((({ int _temp_var_34 = ((values[2] % 4));
        (_temp_var_34 == 0 ? indices.field_0 : (_temp_var_34 == 1 ? indices.field_1 : (_temp_var_34 == 2 ? indices.field_2 : (_temp_var_34 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_33 == 0 ? indices.field_0 : (_temp_var_33 == 1 ? indices.field_1 : (_temp_var_33 == 2 ? indices.field_2 : (_temp_var_33 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_32 == 0 ? indices.field_0 : (_temp_var_32 == 1 ? indices.field_1 : (_temp_var_32 == 2 ? indices.field_2 : (_temp_var_32 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_379(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_382)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_413;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_413 = _block_k_23_(_env_, _kernel_result_382[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_382[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_382[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_382[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_413 = 37;
    }
        
        _result_[_tid_] = temp_stencil_413;
    }
}



// TODO: There should be a better to check if _block_k_25_ is already defined
#ifndef _block_k_25__func
#define _block_k_25__func
__device__ int _block_k_25_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_35 = ((({ int _temp_var_36 = ((({ int _temp_var_37 = ((values[2] % 4));
        (_temp_var_37 == 0 ? indices.field_0 : (_temp_var_37 == 1 ? indices.field_1 : (_temp_var_37 == 2 ? indices.field_2 : (_temp_var_37 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_36 == 0 ? indices.field_0 : (_temp_var_36 == 1 ? indices.field_1 : (_temp_var_36 == 2 ? indices.field_2 : (_temp_var_36 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_35 == 0 ? indices.field_0 : (_temp_var_35 == 1 ? indices.field_1 : (_temp_var_35 == 2 ? indices.field_2 : (_temp_var_35 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_377(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_380)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_414;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_414 = _block_k_25_(_env_, _kernel_result_380[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_380[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_380[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_380[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_414 = 37;
    }
        
        _result_[_tid_] = temp_stencil_414;
    }
}



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_38 = ((({ int _temp_var_39 = ((({ int _temp_var_40 = ((values[2] % 4));
        (_temp_var_40 == 0 ? indices.field_0 : (_temp_var_40 == 1 ? indices.field_1 : (_temp_var_40 == 2 ? indices.field_2 : (_temp_var_40 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_39 == 0 ? indices.field_0 : (_temp_var_39 == 1 ? indices.field_1 : (_temp_var_39 == 2 ? indices.field_2 : (_temp_var_39 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_38 == 0 ? indices.field_0 : (_temp_var_38 == 1 ? indices.field_1 : (_temp_var_38 == 2 ? indices.field_2 : (_temp_var_38 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_375(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_378)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_415;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_415 = _block_k_27_(_env_, _kernel_result_378[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_378[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_378[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_378[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_415 = 37;
    }
        
        _result_[_tid_] = temp_stencil_415;
    }
}



// TODO: There should be a better to check if _block_k_29_ is already defined
#ifndef _block_k_29__func
#define _block_k_29__func
__device__ int _block_k_29_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_41 = ((({ int _temp_var_42 = ((({ int _temp_var_43 = ((values[2] % 4));
        (_temp_var_43 == 0 ? indices.field_0 : (_temp_var_43 == 1 ? indices.field_1 : (_temp_var_43 == 2 ? indices.field_2 : (_temp_var_43 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_42 == 0 ? indices.field_0 : (_temp_var_42 == 1 ? indices.field_1 : (_temp_var_42 == 2 ? indices.field_2 : (_temp_var_42 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_41 == 0 ? indices.field_0 : (_temp_var_41 == 1 ? indices.field_1 : (_temp_var_41 == 2 ? indices.field_2 : (_temp_var_41 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_373(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_376)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_416;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_416 = _block_k_29_(_env_, _kernel_result_376[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_376[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_376[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_376[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_416 = 37;
    }
        
        _result_[_tid_] = temp_stencil_416;
    }
}



// TODO: There should be a better to check if _block_k_31_ is already defined
#ifndef _block_k_31__func
#define _block_k_31__func
__device__ int _block_k_31_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_44 = ((({ int _temp_var_45 = ((({ int _temp_var_46 = ((values[2] % 4));
        (_temp_var_46 == 0 ? indices.field_0 : (_temp_var_46 == 1 ? indices.field_1 : (_temp_var_46 == 2 ? indices.field_2 : (_temp_var_46 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_45 == 0 ? indices.field_0 : (_temp_var_45 == 1 ? indices.field_1 : (_temp_var_45 == 2 ? indices.field_2 : (_temp_var_45 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_44 == 0 ? indices.field_0 : (_temp_var_44 == 1 ? indices.field_1 : (_temp_var_44 == 2 ? indices.field_2 : (_temp_var_44 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_371(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_374)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_417;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_417 = _block_k_31_(_env_, _kernel_result_374[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_374[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_374[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_374[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_417 = 37;
    }
        
        _result_[_tid_] = temp_stencil_417;
    }
}



// TODO: There should be a better to check if _block_k_33_ is already defined
#ifndef _block_k_33__func
#define _block_k_33__func
__device__ int _block_k_33_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_47 = ((({ int _temp_var_48 = ((({ int _temp_var_49 = ((values[2] % 4));
        (_temp_var_49 == 0 ? indices.field_0 : (_temp_var_49 == 1 ? indices.field_1 : (_temp_var_49 == 2 ? indices.field_2 : (_temp_var_49 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_48 == 0 ? indices.field_0 : (_temp_var_48 == 1 ? indices.field_1 : (_temp_var_48 == 2 ? indices.field_2 : (_temp_var_48 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_47 == 0 ? indices.field_0 : (_temp_var_47 == 1 ? indices.field_1 : (_temp_var_47 == 2 ? indices.field_2 : (_temp_var_47 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_369(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_372)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_418;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_418 = _block_k_33_(_env_, _kernel_result_372[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_372[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_372[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_372[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_418 = 37;
    }
        
        _result_[_tid_] = temp_stencil_418;
    }
}



// TODO: There should be a better to check if _block_k_35_ is already defined
#ifndef _block_k_35__func
#define _block_k_35__func
__device__ int _block_k_35_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_50 = ((({ int _temp_var_51 = ((({ int _temp_var_52 = ((values[2] % 4));
        (_temp_var_52 == 0 ? indices.field_0 : (_temp_var_52 == 1 ? indices.field_1 : (_temp_var_52 == 2 ? indices.field_2 : (_temp_var_52 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_51 == 0 ? indices.field_0 : (_temp_var_51 == 1 ? indices.field_1 : (_temp_var_51 == 2 ? indices.field_2 : (_temp_var_51 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_50 == 0 ? indices.field_0 : (_temp_var_50 == 1 ? indices.field_1 : (_temp_var_50 == 2 ? indices.field_2 : (_temp_var_50 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_367(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_370)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_419;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_419 = _block_k_35_(_env_, _kernel_result_370[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_370[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_370[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_370[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_419 = 37;
    }
        
        _result_[_tid_] = temp_stencil_419;
    }
}



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_53 = ((({ int _temp_var_54 = ((({ int _temp_var_55 = ((values[2] % 4));
        (_temp_var_55 == 0 ? indices.field_0 : (_temp_var_55 == 1 ? indices.field_1 : (_temp_var_55 == 2 ? indices.field_2 : (_temp_var_55 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_54 == 0 ? indices.field_0 : (_temp_var_54 == 1 ? indices.field_1 : (_temp_var_54 == 2 ? indices.field_2 : (_temp_var_54 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_53 == 0 ? indices.field_0 : (_temp_var_53 == 1 ? indices.field_1 : (_temp_var_53 == 2 ? indices.field_2 : (_temp_var_53 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_365(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_368)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_420;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_420 = _block_k_37_(_env_, _kernel_result_368[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_368[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_368[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_368[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_420 = 37;
    }
        
        _result_[_tid_] = temp_stencil_420;
    }
}



// TODO: There should be a better to check if _block_k_39_ is already defined
#ifndef _block_k_39__func
#define _block_k_39__func
__device__ int _block_k_39_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_56 = ((({ int _temp_var_57 = ((({ int _temp_var_58 = ((values[2] % 4));
        (_temp_var_58 == 0 ? indices.field_0 : (_temp_var_58 == 1 ? indices.field_1 : (_temp_var_58 == 2 ? indices.field_2 : (_temp_var_58 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_57 == 0 ? indices.field_0 : (_temp_var_57 == 1 ? indices.field_1 : (_temp_var_57 == 2 ? indices.field_2 : (_temp_var_57 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_56 == 0 ? indices.field_0 : (_temp_var_56 == 1 ? indices.field_1 : (_temp_var_56 == 2 ? indices.field_2 : (_temp_var_56 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_363(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_366)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_421;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_421 = _block_k_39_(_env_, _kernel_result_366[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_366[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_366[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_366[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_421 = 37;
    }
        
        _result_[_tid_] = temp_stencil_421;
    }
}



// TODO: There should be a better to check if _block_k_41_ is already defined
#ifndef _block_k_41__func
#define _block_k_41__func
__device__ int _block_k_41_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_59 = ((({ int _temp_var_60 = ((({ int _temp_var_61 = ((values[2] % 4));
        (_temp_var_61 == 0 ? indices.field_0 : (_temp_var_61 == 1 ? indices.field_1 : (_temp_var_61 == 2 ? indices.field_2 : (_temp_var_61 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_60 == 0 ? indices.field_0 : (_temp_var_60 == 1 ? indices.field_1 : (_temp_var_60 == 2 ? indices.field_2 : (_temp_var_60 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_59 == 0 ? indices.field_0 : (_temp_var_59 == 1 ? indices.field_1 : (_temp_var_59 == 2 ? indices.field_2 : (_temp_var_59 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_361(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_364)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_422;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_422 = _block_k_41_(_env_, _kernel_result_364[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_364[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_364[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_364[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_422 = 37;
    }
        
        _result_[_tid_] = temp_stencil_422;
    }
}



// TODO: There should be a better to check if _block_k_43_ is already defined
#ifndef _block_k_43__func
#define _block_k_43__func
__device__ int _block_k_43_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_62 = ((({ int _temp_var_63 = ((({ int _temp_var_64 = ((values[2] % 4));
        (_temp_var_64 == 0 ? indices.field_0 : (_temp_var_64 == 1 ? indices.field_1 : (_temp_var_64 == 2 ? indices.field_2 : (_temp_var_64 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_63 == 0 ? indices.field_0 : (_temp_var_63 == 1 ? indices.field_1 : (_temp_var_63 == 2 ? indices.field_2 : (_temp_var_63 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_62 == 0 ? indices.field_0 : (_temp_var_62 == 1 ? indices.field_1 : (_temp_var_62 == 2 ? indices.field_2 : (_temp_var_62 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_359(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_362)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_423;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_423 = _block_k_43_(_env_, _kernel_result_362[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_362[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_362[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_362[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_423 = 37;
    }
        
        _result_[_tid_] = temp_stencil_423;
    }
}



// TODO: There should be a better to check if _block_k_45_ is already defined
#ifndef _block_k_45__func
#define _block_k_45__func
__device__ int _block_k_45_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_65 = ((({ int _temp_var_66 = ((({ int _temp_var_67 = ((values[2] % 4));
        (_temp_var_67 == 0 ? indices.field_0 : (_temp_var_67 == 1 ? indices.field_1 : (_temp_var_67 == 2 ? indices.field_2 : (_temp_var_67 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_66 == 0 ? indices.field_0 : (_temp_var_66 == 1 ? indices.field_1 : (_temp_var_66 == 2 ? indices.field_2 : (_temp_var_66 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_65 == 0 ? indices.field_0 : (_temp_var_65 == 1 ? indices.field_1 : (_temp_var_65 == 2 ? indices.field_2 : (_temp_var_65 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_357(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_360)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_424;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_424 = _block_k_45_(_env_, _kernel_result_360[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_360[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_360[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_360[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_424 = 37;
    }
        
        _result_[_tid_] = temp_stencil_424;
    }
}



// TODO: There should be a better to check if _block_k_47_ is already defined
#ifndef _block_k_47__func
#define _block_k_47__func
__device__ int _block_k_47_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_68 = ((({ int _temp_var_69 = ((({ int _temp_var_70 = ((values[2] % 4));
        (_temp_var_70 == 0 ? indices.field_0 : (_temp_var_70 == 1 ? indices.field_1 : (_temp_var_70 == 2 ? indices.field_2 : (_temp_var_70 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_69 == 0 ? indices.field_0 : (_temp_var_69 == 1 ? indices.field_1 : (_temp_var_69 == 2 ? indices.field_2 : (_temp_var_69 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_68 == 0 ? indices.field_0 : (_temp_var_68 == 1 ? indices.field_1 : (_temp_var_68 == 2 ? indices.field_2 : (_temp_var_68 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_355(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_358)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_425;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_425 = _block_k_47_(_env_, _kernel_result_358[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_358[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_358[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_358[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_425 = 37;
    }
        
        _result_[_tid_] = temp_stencil_425;
    }
}



// TODO: There should be a better to check if _block_k_49_ is already defined
#ifndef _block_k_49__func
#define _block_k_49__func
__device__ int _block_k_49_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_71 = ((({ int _temp_var_72 = ((({ int _temp_var_73 = ((values[2] % 4));
        (_temp_var_73 == 0 ? indices.field_0 : (_temp_var_73 == 1 ? indices.field_1 : (_temp_var_73 == 2 ? indices.field_2 : (_temp_var_73 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_72 == 0 ? indices.field_0 : (_temp_var_72 == 1 ? indices.field_1 : (_temp_var_72 == 2 ? indices.field_2 : (_temp_var_72 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_71 == 0 ? indices.field_0 : (_temp_var_71 == 1 ? indices.field_1 : (_temp_var_71 == 2 ? indices.field_2 : (_temp_var_71 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_353(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_356)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_426;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_426 = _block_k_49_(_env_, _kernel_result_356[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_356[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_356[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_356[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_426 = 37;
    }
        
        _result_[_tid_] = temp_stencil_426;
    }
}



// TODO: There should be a better to check if _block_k_51_ is already defined
#ifndef _block_k_51__func
#define _block_k_51__func
__device__ int _block_k_51_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_74 = ((({ int _temp_var_75 = ((({ int _temp_var_76 = ((values[2] % 4));
        (_temp_var_76 == 0 ? indices.field_0 : (_temp_var_76 == 1 ? indices.field_1 : (_temp_var_76 == 2 ? indices.field_2 : (_temp_var_76 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_75 == 0 ? indices.field_0 : (_temp_var_75 == 1 ? indices.field_1 : (_temp_var_75 == 2 ? indices.field_2 : (_temp_var_75 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_74 == 0 ? indices.field_0 : (_temp_var_74 == 1 ? indices.field_1 : (_temp_var_74 == 2 ? indices.field_2 : (_temp_var_74 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_351(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_354)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_427;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_427 = _block_k_51_(_env_, _kernel_result_354[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_354[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_354[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_354[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_427 = 37;
    }
        
        _result_[_tid_] = temp_stencil_427;
    }
}



// TODO: There should be a better to check if _block_k_53_ is already defined
#ifndef _block_k_53__func
#define _block_k_53__func
__device__ int _block_k_53_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_77 = ((({ int _temp_var_78 = ((({ int _temp_var_79 = ((values[2] % 4));
        (_temp_var_79 == 0 ? indices.field_0 : (_temp_var_79 == 1 ? indices.field_1 : (_temp_var_79 == 2 ? indices.field_2 : (_temp_var_79 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_78 == 0 ? indices.field_0 : (_temp_var_78 == 1 ? indices.field_1 : (_temp_var_78 == 2 ? indices.field_2 : (_temp_var_78 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_77 == 0 ? indices.field_0 : (_temp_var_77 == 1 ? indices.field_1 : (_temp_var_77 == 2 ? indices.field_2 : (_temp_var_77 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_349(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_352)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_428;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_428 = _block_k_53_(_env_, _kernel_result_352[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_352[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_352[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_352[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_428 = 37;
    }
        
        _result_[_tid_] = temp_stencil_428;
    }
}



// TODO: There should be a better to check if _block_k_55_ is already defined
#ifndef _block_k_55__func
#define _block_k_55__func
__device__ int _block_k_55_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_80 = ((({ int _temp_var_81 = ((({ int _temp_var_82 = ((values[2] % 4));
        (_temp_var_82 == 0 ? indices.field_0 : (_temp_var_82 == 1 ? indices.field_1 : (_temp_var_82 == 2 ? indices.field_2 : (_temp_var_82 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_81 == 0 ? indices.field_0 : (_temp_var_81 == 1 ? indices.field_1 : (_temp_var_81 == 2 ? indices.field_2 : (_temp_var_81 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_80 == 0 ? indices.field_0 : (_temp_var_80 == 1 ? indices.field_1 : (_temp_var_80 == 2 ? indices.field_2 : (_temp_var_80 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_347(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_350)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_429;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_429 = _block_k_55_(_env_, _kernel_result_350[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_350[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_350[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_350[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_429 = 37;
    }
        
        _result_[_tid_] = temp_stencil_429;
    }
}



// TODO: There should be a better to check if _block_k_57_ is already defined
#ifndef _block_k_57__func
#define _block_k_57__func
__device__ int _block_k_57_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_83 = ((({ int _temp_var_84 = ((({ int _temp_var_85 = ((values[2] % 4));
        (_temp_var_85 == 0 ? indices.field_0 : (_temp_var_85 == 1 ? indices.field_1 : (_temp_var_85 == 2 ? indices.field_2 : (_temp_var_85 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_84 == 0 ? indices.field_0 : (_temp_var_84 == 1 ? indices.field_1 : (_temp_var_84 == 2 ? indices.field_2 : (_temp_var_84 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_83 == 0 ? indices.field_0 : (_temp_var_83 == 1 ? indices.field_1 : (_temp_var_83 == 2 ? indices.field_2 : (_temp_var_83 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_345(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_348)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_430;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_430 = _block_k_57_(_env_, _kernel_result_348[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_348[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_348[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_348[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_430 = 37;
    }
        
        _result_[_tid_] = temp_stencil_430;
    }
}



// TODO: There should be a better to check if _block_k_59_ is already defined
#ifndef _block_k_59__func
#define _block_k_59__func
__device__ int _block_k_59_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_86 = ((({ int _temp_var_87 = ((({ int _temp_var_88 = ((values[2] % 4));
        (_temp_var_88 == 0 ? indices.field_0 : (_temp_var_88 == 1 ? indices.field_1 : (_temp_var_88 == 2 ? indices.field_2 : (_temp_var_88 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_87 == 0 ? indices.field_0 : (_temp_var_87 == 1 ? indices.field_1 : (_temp_var_87 == 2 ? indices.field_2 : (_temp_var_87 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_86 == 0 ? indices.field_0 : (_temp_var_86 == 1 ? indices.field_1 : (_temp_var_86 == 2 ? indices.field_2 : (_temp_var_86 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_343(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_346)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_431;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_431 = _block_k_59_(_env_, _kernel_result_346[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_346[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_346[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_346[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_431 = 37;
    }
        
        _result_[_tid_] = temp_stencil_431;
    }
}



// TODO: There should be a better to check if _block_k_61_ is already defined
#ifndef _block_k_61__func
#define _block_k_61__func
__device__ int _block_k_61_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_89 = ((({ int _temp_var_90 = ((({ int _temp_var_91 = ((values[2] % 4));
        (_temp_var_91 == 0 ? indices.field_0 : (_temp_var_91 == 1 ? indices.field_1 : (_temp_var_91 == 2 ? indices.field_2 : (_temp_var_91 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_90 == 0 ? indices.field_0 : (_temp_var_90 == 1 ? indices.field_1 : (_temp_var_90 == 2 ? indices.field_2 : (_temp_var_90 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_89 == 0 ? indices.field_0 : (_temp_var_89 == 1 ? indices.field_1 : (_temp_var_89 == 2 ? indices.field_2 : (_temp_var_89 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_341(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_344)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_432;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_432 = _block_k_61_(_env_, _kernel_result_344[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_344[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_344[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_344[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_432 = 37;
    }
        
        _result_[_tid_] = temp_stencil_432;
    }
}



// TODO: There should be a better to check if _block_k_63_ is already defined
#ifndef _block_k_63__func
#define _block_k_63__func
__device__ int _block_k_63_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_92 = ((({ int _temp_var_93 = ((({ int _temp_var_94 = ((values[2] % 4));
        (_temp_var_94 == 0 ? indices.field_0 : (_temp_var_94 == 1 ? indices.field_1 : (_temp_var_94 == 2 ? indices.field_2 : (_temp_var_94 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_93 == 0 ? indices.field_0 : (_temp_var_93 == 1 ? indices.field_1 : (_temp_var_93 == 2 ? indices.field_2 : (_temp_var_93 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_92 == 0 ? indices.field_0 : (_temp_var_92 == 1 ? indices.field_1 : (_temp_var_92 == 2 ? indices.field_2 : (_temp_var_92 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_339(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_342)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_433;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_433 = _block_k_63_(_env_, _kernel_result_342[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_342[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_342[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_342[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_433 = 37;
    }
        
        _result_[_tid_] = temp_stencil_433;
    }
}



// TODO: There should be a better to check if _block_k_65_ is already defined
#ifndef _block_k_65__func
#define _block_k_65__func
__device__ int _block_k_65_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_95 = ((({ int _temp_var_96 = ((({ int _temp_var_97 = ((values[2] % 4));
        (_temp_var_97 == 0 ? indices.field_0 : (_temp_var_97 == 1 ? indices.field_1 : (_temp_var_97 == 2 ? indices.field_2 : (_temp_var_97 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_96 == 0 ? indices.field_0 : (_temp_var_96 == 1 ? indices.field_1 : (_temp_var_96 == 2 ? indices.field_2 : (_temp_var_96 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_95 == 0 ? indices.field_0 : (_temp_var_95 == 1 ? indices.field_1 : (_temp_var_95 == 2 ? indices.field_2 : (_temp_var_95 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_337(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_340)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_434;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_434 = _block_k_65_(_env_, _kernel_result_340[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_340[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_340[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_340[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_434 = 37;
    }
        
        _result_[_tid_] = temp_stencil_434;
    }
}



// TODO: There should be a better to check if _block_k_67_ is already defined
#ifndef _block_k_67__func
#define _block_k_67__func
__device__ int _block_k_67_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_98 = ((({ int _temp_var_99 = ((({ int _temp_var_100 = ((values[2] % 4));
        (_temp_var_100 == 0 ? indices.field_0 : (_temp_var_100 == 1 ? indices.field_1 : (_temp_var_100 == 2 ? indices.field_2 : (_temp_var_100 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_99 == 0 ? indices.field_0 : (_temp_var_99 == 1 ? indices.field_1 : (_temp_var_99 == 2 ? indices.field_2 : (_temp_var_99 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_98 == 0 ? indices.field_0 : (_temp_var_98 == 1 ? indices.field_1 : (_temp_var_98 == 2 ? indices.field_2 : (_temp_var_98 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_335(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_338)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_435;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_435 = _block_k_67_(_env_, _kernel_result_338[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_338[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_338[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_338[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_435 = 37;
    }
        
        _result_[_tid_] = temp_stencil_435;
    }
}



// TODO: There should be a better to check if _block_k_69_ is already defined
#ifndef _block_k_69__func
#define _block_k_69__func
__device__ int _block_k_69_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_101 = ((({ int _temp_var_102 = ((({ int _temp_var_103 = ((values[2] % 4));
        (_temp_var_103 == 0 ? indices.field_0 : (_temp_var_103 == 1 ? indices.field_1 : (_temp_var_103 == 2 ? indices.field_2 : (_temp_var_103 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_102 == 0 ? indices.field_0 : (_temp_var_102 == 1 ? indices.field_1 : (_temp_var_102 == 2 ? indices.field_2 : (_temp_var_102 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_101 == 0 ? indices.field_0 : (_temp_var_101 == 1 ? indices.field_1 : (_temp_var_101 == 2 ? indices.field_2 : (_temp_var_101 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_333(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_336)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_436;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_436 = _block_k_69_(_env_, _kernel_result_336[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_336[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_336[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_336[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_436 = 37;
    }
        
        _result_[_tid_] = temp_stencil_436;
    }
}



// TODO: There should be a better to check if _block_k_71_ is already defined
#ifndef _block_k_71__func
#define _block_k_71__func
__device__ int _block_k_71_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_104 = ((({ int _temp_var_105 = ((({ int _temp_var_106 = ((values[2] % 4));
        (_temp_var_106 == 0 ? indices.field_0 : (_temp_var_106 == 1 ? indices.field_1 : (_temp_var_106 == 2 ? indices.field_2 : (_temp_var_106 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_105 == 0 ? indices.field_0 : (_temp_var_105 == 1 ? indices.field_1 : (_temp_var_105 == 2 ? indices.field_2 : (_temp_var_105 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_104 == 0 ? indices.field_0 : (_temp_var_104 == 1 ? indices.field_1 : (_temp_var_104 == 2 ? indices.field_2 : (_temp_var_104 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_331(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_334)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_437;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_437 = _block_k_71_(_env_, _kernel_result_334[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_334[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_334[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_334[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_437 = 37;
    }
        
        _result_[_tid_] = temp_stencil_437;
    }
}



// TODO: There should be a better to check if _block_k_73_ is already defined
#ifndef _block_k_73__func
#define _block_k_73__func
__device__ int _block_k_73_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_107 = ((({ int _temp_var_108 = ((({ int _temp_var_109 = ((values[2] % 4));
        (_temp_var_109 == 0 ? indices.field_0 : (_temp_var_109 == 1 ? indices.field_1 : (_temp_var_109 == 2 ? indices.field_2 : (_temp_var_109 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_108 == 0 ? indices.field_0 : (_temp_var_108 == 1 ? indices.field_1 : (_temp_var_108 == 2 ? indices.field_2 : (_temp_var_108 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_107 == 0 ? indices.field_0 : (_temp_var_107 == 1 ? indices.field_1 : (_temp_var_107 == 2 ? indices.field_2 : (_temp_var_107 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_329(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_332)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_438;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_438 = _block_k_73_(_env_, _kernel_result_332[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_332[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_332[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_332[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_438 = 37;
    }
        
        _result_[_tid_] = temp_stencil_438;
    }
}



// TODO: There should be a better to check if _block_k_75_ is already defined
#ifndef _block_k_75__func
#define _block_k_75__func
__device__ int _block_k_75_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_110 = ((({ int _temp_var_111 = ((({ int _temp_var_112 = ((values[2] % 4));
        (_temp_var_112 == 0 ? indices.field_0 : (_temp_var_112 == 1 ? indices.field_1 : (_temp_var_112 == 2 ? indices.field_2 : (_temp_var_112 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_111 == 0 ? indices.field_0 : (_temp_var_111 == 1 ? indices.field_1 : (_temp_var_111 == 2 ? indices.field_2 : (_temp_var_111 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_110 == 0 ? indices.field_0 : (_temp_var_110 == 1 ? indices.field_1 : (_temp_var_110 == 2 ? indices.field_2 : (_temp_var_110 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_327(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_330)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_439;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_439 = _block_k_75_(_env_, _kernel_result_330[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_330[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_330[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_330[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_439 = 37;
    }
        
        _result_[_tid_] = temp_stencil_439;
    }
}



// TODO: There should be a better to check if _block_k_77_ is already defined
#ifndef _block_k_77__func
#define _block_k_77__func
__device__ int _block_k_77_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_113 = ((({ int _temp_var_114 = ((({ int _temp_var_115 = ((values[2] % 4));
        (_temp_var_115 == 0 ? indices.field_0 : (_temp_var_115 == 1 ? indices.field_1 : (_temp_var_115 == 2 ? indices.field_2 : (_temp_var_115 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_114 == 0 ? indices.field_0 : (_temp_var_114 == 1 ? indices.field_1 : (_temp_var_114 == 2 ? indices.field_2 : (_temp_var_114 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_113 == 0 ? indices.field_0 : (_temp_var_113 == 1 ? indices.field_1 : (_temp_var_113 == 2 ? indices.field_2 : (_temp_var_113 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_325(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_328)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_440;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_440 = _block_k_77_(_env_, _kernel_result_328[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_328[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_328[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_328[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_440 = 37;
    }
        
        _result_[_tid_] = temp_stencil_440;
    }
}



// TODO: There should be a better to check if _block_k_79_ is already defined
#ifndef _block_k_79__func
#define _block_k_79__func
__device__ int _block_k_79_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_116 = ((({ int _temp_var_117 = ((({ int _temp_var_118 = ((values[2] % 4));
        (_temp_var_118 == 0 ? indices.field_0 : (_temp_var_118 == 1 ? indices.field_1 : (_temp_var_118 == 2 ? indices.field_2 : (_temp_var_118 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_117 == 0 ? indices.field_0 : (_temp_var_117 == 1 ? indices.field_1 : (_temp_var_117 == 2 ? indices.field_2 : (_temp_var_117 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_116 == 0 ? indices.field_0 : (_temp_var_116 == 1 ? indices.field_1 : (_temp_var_116 == 2 ? indices.field_2 : (_temp_var_116 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_323(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_326)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_441;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_441 = _block_k_79_(_env_, _kernel_result_326[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_326[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_326[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_326[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_441 = 37;
    }
        
        _result_[_tid_] = temp_stencil_441;
    }
}



// TODO: There should be a better to check if _block_k_81_ is already defined
#ifndef _block_k_81__func
#define _block_k_81__func
__device__ int _block_k_81_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_119 = ((({ int _temp_var_120 = ((({ int _temp_var_121 = ((values[2] % 4));
        (_temp_var_121 == 0 ? indices.field_0 : (_temp_var_121 == 1 ? indices.field_1 : (_temp_var_121 == 2 ? indices.field_2 : (_temp_var_121 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_120 == 0 ? indices.field_0 : (_temp_var_120 == 1 ? indices.field_1 : (_temp_var_120 == 2 ? indices.field_2 : (_temp_var_120 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_119 == 0 ? indices.field_0 : (_temp_var_119 == 1 ? indices.field_1 : (_temp_var_119 == 2 ? indices.field_2 : (_temp_var_119 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_321(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_324)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_442;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_442 = _block_k_81_(_env_, _kernel_result_324[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_324[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_324[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_324[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_442 = 37;
    }
        
        _result_[_tid_] = temp_stencil_442;
    }
}



// TODO: There should be a better to check if _block_k_83_ is already defined
#ifndef _block_k_83__func
#define _block_k_83__func
__device__ int _block_k_83_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_122 = ((({ int _temp_var_123 = ((({ int _temp_var_124 = ((values[2] % 4));
        (_temp_var_124 == 0 ? indices.field_0 : (_temp_var_124 == 1 ? indices.field_1 : (_temp_var_124 == 2 ? indices.field_2 : (_temp_var_124 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_123 == 0 ? indices.field_0 : (_temp_var_123 == 1 ? indices.field_1 : (_temp_var_123 == 2 ? indices.field_2 : (_temp_var_123 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_122 == 0 ? indices.field_0 : (_temp_var_122 == 1 ? indices.field_1 : (_temp_var_122 == 2 ? indices.field_2 : (_temp_var_122 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_319(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_322)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_443;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_443 = _block_k_83_(_env_, _kernel_result_322[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_322[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_322[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_322[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_443 = 37;
    }
        
        _result_[_tid_] = temp_stencil_443;
    }
}



// TODO: There should be a better to check if _block_k_85_ is already defined
#ifndef _block_k_85__func
#define _block_k_85__func
__device__ int _block_k_85_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_125 = ((({ int _temp_var_126 = ((({ int _temp_var_127 = ((values[2] % 4));
        (_temp_var_127 == 0 ? indices.field_0 : (_temp_var_127 == 1 ? indices.field_1 : (_temp_var_127 == 2 ? indices.field_2 : (_temp_var_127 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_126 == 0 ? indices.field_0 : (_temp_var_126 == 1 ? indices.field_1 : (_temp_var_126 == 2 ? indices.field_2 : (_temp_var_126 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_125 == 0 ? indices.field_0 : (_temp_var_125 == 1 ? indices.field_1 : (_temp_var_125 == 2 ? indices.field_2 : (_temp_var_125 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_317(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_320)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_444;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_444 = _block_k_85_(_env_, _kernel_result_320[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_320[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_320[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_320[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_444 = 37;
    }
        
        _result_[_tid_] = temp_stencil_444;
    }
}



// TODO: There should be a better to check if _block_k_87_ is already defined
#ifndef _block_k_87__func
#define _block_k_87__func
__device__ int _block_k_87_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_128 = ((({ int _temp_var_129 = ((({ int _temp_var_130 = ((values[2] % 4));
        (_temp_var_130 == 0 ? indices.field_0 : (_temp_var_130 == 1 ? indices.field_1 : (_temp_var_130 == 2 ? indices.field_2 : (_temp_var_130 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_129 == 0 ? indices.field_0 : (_temp_var_129 == 1 ? indices.field_1 : (_temp_var_129 == 2 ? indices.field_2 : (_temp_var_129 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_128 == 0 ? indices.field_0 : (_temp_var_128 == 1 ? indices.field_1 : (_temp_var_128 == 2 ? indices.field_2 : (_temp_var_128 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_315(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_318)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_445;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_445 = _block_k_87_(_env_, _kernel_result_318[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_318[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_318[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_318[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_445 = 37;
    }
        
        _result_[_tid_] = temp_stencil_445;
    }
}



// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_131 = ((({ int _temp_var_132 = ((({ int _temp_var_133 = ((values[2] % 4));
        (_temp_var_133 == 0 ? indices.field_0 : (_temp_var_133 == 1 ? indices.field_1 : (_temp_var_133 == 2 ? indices.field_2 : (_temp_var_133 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_132 == 0 ? indices.field_0 : (_temp_var_132 == 1 ? indices.field_1 : (_temp_var_132 == 2 ? indices.field_2 : (_temp_var_132 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_131 == 0 ? indices.field_0 : (_temp_var_131 == 1 ? indices.field_1 : (_temp_var_131 == 2 ? indices.field_2 : (_temp_var_131 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_313(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_316)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_446;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_446 = _block_k_89_(_env_, _kernel_result_316[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_316[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_316[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_316[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_446 = 37;
    }
        
        _result_[_tid_] = temp_stencil_446;
    }
}



// TODO: There should be a better to check if _block_k_91_ is already defined
#ifndef _block_k_91__func
#define _block_k_91__func
__device__ int _block_k_91_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_134 = ((({ int _temp_var_135 = ((({ int _temp_var_136 = ((values[2] % 4));
        (_temp_var_136 == 0 ? indices.field_0 : (_temp_var_136 == 1 ? indices.field_1 : (_temp_var_136 == 2 ? indices.field_2 : (_temp_var_136 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_135 == 0 ? indices.field_0 : (_temp_var_135 == 1 ? indices.field_1 : (_temp_var_135 == 2 ? indices.field_2 : (_temp_var_135 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_134 == 0 ? indices.field_0 : (_temp_var_134 == 1 ? indices.field_1 : (_temp_var_134 == 2 ? indices.field_2 : (_temp_var_134 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_311(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_314)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_447;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_447 = _block_k_91_(_env_, _kernel_result_314[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_314[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_314[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_314[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_447 = 37;
    }
        
        _result_[_tid_] = temp_stencil_447;
    }
}



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_137 = ((({ int _temp_var_138 = ((({ int _temp_var_139 = ((values[2] % 4));
        (_temp_var_139 == 0 ? indices.field_0 : (_temp_var_139 == 1 ? indices.field_1 : (_temp_var_139 == 2 ? indices.field_2 : (_temp_var_139 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_138 == 0 ? indices.field_0 : (_temp_var_138 == 1 ? indices.field_1 : (_temp_var_138 == 2 ? indices.field_2 : (_temp_var_138 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_137 == 0 ? indices.field_0 : (_temp_var_137 == 1 ? indices.field_1 : (_temp_var_137 == 2 ? indices.field_2 : (_temp_var_137 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_309(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_312)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_448;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_448 = _block_k_93_(_env_, _kernel_result_312[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_312[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_312[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_312[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_448 = 37;
    }
        
        _result_[_tid_] = temp_stencil_448;
    }
}



// TODO: There should be a better to check if _block_k_95_ is already defined
#ifndef _block_k_95__func
#define _block_k_95__func
__device__ int _block_k_95_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_140 = ((({ int _temp_var_141 = ((({ int _temp_var_142 = ((values[2] % 4));
        (_temp_var_142 == 0 ? indices.field_0 : (_temp_var_142 == 1 ? indices.field_1 : (_temp_var_142 == 2 ? indices.field_2 : (_temp_var_142 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_141 == 0 ? indices.field_0 : (_temp_var_141 == 1 ? indices.field_1 : (_temp_var_141 == 2 ? indices.field_2 : (_temp_var_141 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_140 == 0 ? indices.field_0 : (_temp_var_140 == 1 ? indices.field_1 : (_temp_var_140 == 2 ? indices.field_2 : (_temp_var_140 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_307(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_310)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_449;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_449 = _block_k_95_(_env_, _kernel_result_310[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_310[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_310[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_310[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_449 = 37;
    }
        
        _result_[_tid_] = temp_stencil_449;
    }
}



// TODO: There should be a better to check if _block_k_97_ is already defined
#ifndef _block_k_97__func
#define _block_k_97__func
__device__ int _block_k_97_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_143 = ((({ int _temp_var_144 = ((({ int _temp_var_145 = ((values[2] % 4));
        (_temp_var_145 == 0 ? indices.field_0 : (_temp_var_145 == 1 ? indices.field_1 : (_temp_var_145 == 2 ? indices.field_2 : (_temp_var_145 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_144 == 0 ? indices.field_0 : (_temp_var_144 == 1 ? indices.field_1 : (_temp_var_144 == 2 ? indices.field_2 : (_temp_var_144 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_143 == 0 ? indices.field_0 : (_temp_var_143 == 1 ? indices.field_1 : (_temp_var_143 == 2 ? indices.field_2 : (_temp_var_143 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_305(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_308)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_450;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_450 = _block_k_97_(_env_, _kernel_result_308[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_308[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_308[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_308[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_450 = 37;
    }
        
        _result_[_tid_] = temp_stencil_450;
    }
}



// TODO: There should be a better to check if _block_k_99_ is already defined
#ifndef _block_k_99__func
#define _block_k_99__func
__device__ int _block_k_99_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_146 = ((({ int _temp_var_147 = ((({ int _temp_var_148 = ((values[2] % 4));
        (_temp_var_148 == 0 ? indices.field_0 : (_temp_var_148 == 1 ? indices.field_1 : (_temp_var_148 == 2 ? indices.field_2 : (_temp_var_148 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_147 == 0 ? indices.field_0 : (_temp_var_147 == 1 ? indices.field_1 : (_temp_var_147 == 2 ? indices.field_2 : (_temp_var_147 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_146 == 0 ? indices.field_0 : (_temp_var_146 == 1 ? indices.field_1 : (_temp_var_146 == 2 ? indices.field_2 : (_temp_var_146 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_303(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_306)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_451;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_451 = _block_k_99_(_env_, _kernel_result_306[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_306[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_306[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_306[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_451 = 37;
    }
        
        _result_[_tid_] = temp_stencil_451;
    }
}



// TODO: There should be a better to check if _block_k_101_ is already defined
#ifndef _block_k_101__func
#define _block_k_101__func
__device__ int _block_k_101_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_149 = ((({ int _temp_var_150 = ((({ int _temp_var_151 = ((values[2] % 4));
        (_temp_var_151 == 0 ? indices.field_0 : (_temp_var_151 == 1 ? indices.field_1 : (_temp_var_151 == 2 ? indices.field_2 : (_temp_var_151 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_150 == 0 ? indices.field_0 : (_temp_var_150 == 1 ? indices.field_1 : (_temp_var_150 == 2 ? indices.field_2 : (_temp_var_150 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_149 == 0 ? indices.field_0 : (_temp_var_149 == 1 ? indices.field_1 : (_temp_var_149 == 2 ? indices.field_2 : (_temp_var_149 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_301(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_304)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_452;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_452 = _block_k_101_(_env_, _kernel_result_304[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_304[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_304[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_304[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_452 = 37;
    }
        
        _result_[_tid_] = temp_stencil_452;
    }
}



// TODO: There should be a better to check if _block_k_103_ is already defined
#ifndef _block_k_103__func
#define _block_k_103__func
__device__ int _block_k_103_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_152 = ((({ int _temp_var_153 = ((({ int _temp_var_154 = ((values[2] % 4));
        (_temp_var_154 == 0 ? indices.field_0 : (_temp_var_154 == 1 ? indices.field_1 : (_temp_var_154 == 2 ? indices.field_2 : (_temp_var_154 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_153 == 0 ? indices.field_0 : (_temp_var_153 == 1 ? indices.field_1 : (_temp_var_153 == 2 ? indices.field_2 : (_temp_var_153 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_152 == 0 ? indices.field_0 : (_temp_var_152 == 1 ? indices.field_1 : (_temp_var_152 == 2 ? indices.field_2 : (_temp_var_152 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_299(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_302)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_453;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_453 = _block_k_103_(_env_, _kernel_result_302[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_302[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_302[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_302[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_453 = 37;
    }
        
        _result_[_tid_] = temp_stencil_453;
    }
}



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_155 = ((({ int _temp_var_156 = ((({ int _temp_var_157 = ((values[2] % 4));
        (_temp_var_157 == 0 ? indices.field_0 : (_temp_var_157 == 1 ? indices.field_1 : (_temp_var_157 == 2 ? indices.field_2 : (_temp_var_157 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_156 == 0 ? indices.field_0 : (_temp_var_156 == 1 ? indices.field_1 : (_temp_var_156 == 2 ? indices.field_2 : (_temp_var_156 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_155 == 0 ? indices.field_0 : (_temp_var_155 == 1 ? indices.field_1 : (_temp_var_155 == 2 ? indices.field_2 : (_temp_var_155 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_297(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_300)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_454;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_454 = _block_k_105_(_env_, _kernel_result_300[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_300[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_300[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_300[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_454 = 37;
    }
        
        _result_[_tid_] = temp_stencil_454;
    }
}



// TODO: There should be a better to check if _block_k_107_ is already defined
#ifndef _block_k_107__func
#define _block_k_107__func
__device__ int _block_k_107_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_158 = ((({ int _temp_var_159 = ((({ int _temp_var_160 = ((values[2] % 4));
        (_temp_var_160 == 0 ? indices.field_0 : (_temp_var_160 == 1 ? indices.field_1 : (_temp_var_160 == 2 ? indices.field_2 : (_temp_var_160 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_159 == 0 ? indices.field_0 : (_temp_var_159 == 1 ? indices.field_1 : (_temp_var_159 == 2 ? indices.field_2 : (_temp_var_159 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_158 == 0 ? indices.field_0 : (_temp_var_158 == 1 ? indices.field_1 : (_temp_var_158 == 2 ? indices.field_2 : (_temp_var_158 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_295(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_298)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_455;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_455 = _block_k_107_(_env_, _kernel_result_298[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_298[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_298[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_298[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_455 = 37;
    }
        
        _result_[_tid_] = temp_stencil_455;
    }
}



// TODO: There should be a better to check if _block_k_109_ is already defined
#ifndef _block_k_109__func
#define _block_k_109__func
__device__ int _block_k_109_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_161 = ((({ int _temp_var_162 = ((({ int _temp_var_163 = ((values[2] % 4));
        (_temp_var_163 == 0 ? indices.field_0 : (_temp_var_163 == 1 ? indices.field_1 : (_temp_var_163 == 2 ? indices.field_2 : (_temp_var_163 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_162 == 0 ? indices.field_0 : (_temp_var_162 == 1 ? indices.field_1 : (_temp_var_162 == 2 ? indices.field_2 : (_temp_var_162 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_161 == 0 ? indices.field_0 : (_temp_var_161 == 1 ? indices.field_1 : (_temp_var_161 == 2 ? indices.field_2 : (_temp_var_161 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_293(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_296)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_456;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_456 = _block_k_109_(_env_, _kernel_result_296[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_296[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_296[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_296[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_456 = 37;
    }
        
        _result_[_tid_] = temp_stencil_456;
    }
}



// TODO: There should be a better to check if _block_k_111_ is already defined
#ifndef _block_k_111__func
#define _block_k_111__func
__device__ int _block_k_111_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_164 = ((({ int _temp_var_165 = ((({ int _temp_var_166 = ((values[2] % 4));
        (_temp_var_166 == 0 ? indices.field_0 : (_temp_var_166 == 1 ? indices.field_1 : (_temp_var_166 == 2 ? indices.field_2 : (_temp_var_166 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_165 == 0 ? indices.field_0 : (_temp_var_165 == 1 ? indices.field_1 : (_temp_var_165 == 2 ? indices.field_2 : (_temp_var_165 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_164 == 0 ? indices.field_0 : (_temp_var_164 == 1 ? indices.field_1 : (_temp_var_164 == 2 ? indices.field_2 : (_temp_var_164 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_291(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_294)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_457;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_457 = _block_k_111_(_env_, _kernel_result_294[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_294[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_294[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_294[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_457 = 37;
    }
        
        _result_[_tid_] = temp_stencil_457;
    }
}



// TODO: There should be a better to check if _block_k_113_ is already defined
#ifndef _block_k_113__func
#define _block_k_113__func
__device__ int _block_k_113_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_167 = ((({ int _temp_var_168 = ((({ int _temp_var_169 = ((values[2] % 4));
        (_temp_var_169 == 0 ? indices.field_0 : (_temp_var_169 == 1 ? indices.field_1 : (_temp_var_169 == 2 ? indices.field_2 : (_temp_var_169 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_168 == 0 ? indices.field_0 : (_temp_var_168 == 1 ? indices.field_1 : (_temp_var_168 == 2 ? indices.field_2 : (_temp_var_168 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_167 == 0 ? indices.field_0 : (_temp_var_167 == 1 ? indices.field_1 : (_temp_var_167 == 2 ? indices.field_2 : (_temp_var_167 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_289(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_292)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_458;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_458 = _block_k_113_(_env_, _kernel_result_292[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_292[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_292[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_292[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_458 = 37;
    }
        
        _result_[_tid_] = temp_stencil_458;
    }
}



// TODO: There should be a better to check if _block_k_115_ is already defined
#ifndef _block_k_115__func
#define _block_k_115__func
__device__ int _block_k_115_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_170 = ((({ int _temp_var_171 = ((({ int _temp_var_172 = ((values[2] % 4));
        (_temp_var_172 == 0 ? indices.field_0 : (_temp_var_172 == 1 ? indices.field_1 : (_temp_var_172 == 2 ? indices.field_2 : (_temp_var_172 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_171 == 0 ? indices.field_0 : (_temp_var_171 == 1 ? indices.field_1 : (_temp_var_171 == 2 ? indices.field_2 : (_temp_var_171 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_170 == 0 ? indices.field_0 : (_temp_var_170 == 1 ? indices.field_1 : (_temp_var_170 == 2 ? indices.field_2 : (_temp_var_170 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_287(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_290)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_459;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_459 = _block_k_115_(_env_, _kernel_result_290[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_290[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_290[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_290[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_459 = 37;
    }
        
        _result_[_tid_] = temp_stencil_459;
    }
}



// TODO: There should be a better to check if _block_k_117_ is already defined
#ifndef _block_k_117__func
#define _block_k_117__func
__device__ int _block_k_117_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_173 = ((({ int _temp_var_174 = ((({ int _temp_var_175 = ((values[2] % 4));
        (_temp_var_175 == 0 ? indices.field_0 : (_temp_var_175 == 1 ? indices.field_1 : (_temp_var_175 == 2 ? indices.field_2 : (_temp_var_175 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_174 == 0 ? indices.field_0 : (_temp_var_174 == 1 ? indices.field_1 : (_temp_var_174 == 2 ? indices.field_2 : (_temp_var_174 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_173 == 0 ? indices.field_0 : (_temp_var_173 == 1 ? indices.field_1 : (_temp_var_173 == 2 ? indices.field_2 : (_temp_var_173 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_285(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_288)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_460;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_460 = _block_k_117_(_env_, _kernel_result_288[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_288[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_288[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_288[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_460 = 37;
    }
        
        _result_[_tid_] = temp_stencil_460;
    }
}



// TODO: There should be a better to check if _block_k_119_ is already defined
#ifndef _block_k_119__func
#define _block_k_119__func
__device__ int _block_k_119_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_176 = ((({ int _temp_var_177 = ((({ int _temp_var_178 = ((values[2] % 4));
        (_temp_var_178 == 0 ? indices.field_0 : (_temp_var_178 == 1 ? indices.field_1 : (_temp_var_178 == 2 ? indices.field_2 : (_temp_var_178 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_177 == 0 ? indices.field_0 : (_temp_var_177 == 1 ? indices.field_1 : (_temp_var_177 == 2 ? indices.field_2 : (_temp_var_177 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_176 == 0 ? indices.field_0 : (_temp_var_176 == 1 ? indices.field_1 : (_temp_var_176 == 2 ? indices.field_2 : (_temp_var_176 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_283(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_286)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_461;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_461 = _block_k_119_(_env_, _kernel_result_286[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_286[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_286[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_286[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_461 = 37;
    }
        
        _result_[_tid_] = temp_stencil_461;
    }
}



// TODO: There should be a better to check if _block_k_121_ is already defined
#ifndef _block_k_121__func
#define _block_k_121__func
__device__ int _block_k_121_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_179 = ((({ int _temp_var_180 = ((({ int _temp_var_181 = ((values[2] % 4));
        (_temp_var_181 == 0 ? indices.field_0 : (_temp_var_181 == 1 ? indices.field_1 : (_temp_var_181 == 2 ? indices.field_2 : (_temp_var_181 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_180 == 0 ? indices.field_0 : (_temp_var_180 == 1 ? indices.field_1 : (_temp_var_180 == 2 ? indices.field_2 : (_temp_var_180 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_179 == 0 ? indices.field_0 : (_temp_var_179 == 1 ? indices.field_1 : (_temp_var_179 == 2 ? indices.field_2 : (_temp_var_179 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_281(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_284)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_462;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_462 = _block_k_121_(_env_, _kernel_result_284[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_284[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_284[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_284[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_462 = 37;
    }
        
        _result_[_tid_] = temp_stencil_462;
    }
}



// TODO: There should be a better to check if _block_k_123_ is already defined
#ifndef _block_k_123__func
#define _block_k_123__func
__device__ int _block_k_123_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_182 = ((({ int _temp_var_183 = ((({ int _temp_var_184 = ((values[2] % 4));
        (_temp_var_184 == 0 ? indices.field_0 : (_temp_var_184 == 1 ? indices.field_1 : (_temp_var_184 == 2 ? indices.field_2 : (_temp_var_184 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_183 == 0 ? indices.field_0 : (_temp_var_183 == 1 ? indices.field_1 : (_temp_var_183 == 2 ? indices.field_2 : (_temp_var_183 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_182 == 0 ? indices.field_0 : (_temp_var_182 == 1 ? indices.field_1 : (_temp_var_182 == 2 ? indices.field_2 : (_temp_var_182 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_279(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_282)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_463;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_463 = _block_k_123_(_env_, _kernel_result_282[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_282[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_282[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_282[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_463 = 37;
    }
        
        _result_[_tid_] = temp_stencil_463;
    }
}



// TODO: There should be a better to check if _block_k_125_ is already defined
#ifndef _block_k_125__func
#define _block_k_125__func
__device__ int _block_k_125_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_185 = ((({ int _temp_var_186 = ((({ int _temp_var_187 = ((values[2] % 4));
        (_temp_var_187 == 0 ? indices.field_0 : (_temp_var_187 == 1 ? indices.field_1 : (_temp_var_187 == 2 ? indices.field_2 : (_temp_var_187 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_186 == 0 ? indices.field_0 : (_temp_var_186 == 1 ? indices.field_1 : (_temp_var_186 == 2 ? indices.field_2 : (_temp_var_186 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_185 == 0 ? indices.field_0 : (_temp_var_185 == 1 ? indices.field_1 : (_temp_var_185 == 2 ? indices.field_2 : (_temp_var_185 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_277(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_280)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_464;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_464 = _block_k_125_(_env_, _kernel_result_280[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_280[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_280[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_280[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_464 = 37;
    }
        
        _result_[_tid_] = temp_stencil_464;
    }
}



// TODO: There should be a better to check if _block_k_127_ is already defined
#ifndef _block_k_127__func
#define _block_k_127__func
__device__ int _block_k_127_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_188 = ((({ int _temp_var_189 = ((({ int _temp_var_190 = ((values[2] % 4));
        (_temp_var_190 == 0 ? indices.field_0 : (_temp_var_190 == 1 ? indices.field_1 : (_temp_var_190 == 2 ? indices.field_2 : (_temp_var_190 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_189 == 0 ? indices.field_0 : (_temp_var_189 == 1 ? indices.field_1 : (_temp_var_189 == 2 ? indices.field_2 : (_temp_var_189 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_188 == 0 ? indices.field_0 : (_temp_var_188 == 1 ? indices.field_1 : (_temp_var_188 == 2 ? indices.field_2 : (_temp_var_188 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_275(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_278)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_465;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_465 = _block_k_127_(_env_, _kernel_result_278[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_278[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_278[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_278[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_465 = 37;
    }
        
        _result_[_tid_] = temp_stencil_465;
    }
}



// TODO: There should be a better to check if _block_k_129_ is already defined
#ifndef _block_k_129__func
#define _block_k_129__func
__device__ int _block_k_129_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_191 = ((({ int _temp_var_192 = ((({ int _temp_var_193 = ((values[2] % 4));
        (_temp_var_193 == 0 ? indices.field_0 : (_temp_var_193 == 1 ? indices.field_1 : (_temp_var_193 == 2 ? indices.field_2 : (_temp_var_193 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_192 == 0 ? indices.field_0 : (_temp_var_192 == 1 ? indices.field_1 : (_temp_var_192 == 2 ? indices.field_2 : (_temp_var_192 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_191 == 0 ? indices.field_0 : (_temp_var_191 == 1 ? indices.field_1 : (_temp_var_191 == 2 ? indices.field_2 : (_temp_var_191 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_273(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_276)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_466;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_466 = _block_k_129_(_env_, _kernel_result_276[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_276[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_276[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_276[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_466 = 37;
    }
        
        _result_[_tid_] = temp_stencil_466;
    }
}



// TODO: There should be a better to check if _block_k_131_ is already defined
#ifndef _block_k_131__func
#define _block_k_131__func
__device__ int _block_k_131_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_194 = ((({ int _temp_var_195 = ((({ int _temp_var_196 = ((values[2] % 4));
        (_temp_var_196 == 0 ? indices.field_0 : (_temp_var_196 == 1 ? indices.field_1 : (_temp_var_196 == 2 ? indices.field_2 : (_temp_var_196 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_195 == 0 ? indices.field_0 : (_temp_var_195 == 1 ? indices.field_1 : (_temp_var_195 == 2 ? indices.field_2 : (_temp_var_195 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_194 == 0 ? indices.field_0 : (_temp_var_194 == 1 ? indices.field_1 : (_temp_var_194 == 2 ? indices.field_2 : (_temp_var_194 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_271(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_274)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_467;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_467 = _block_k_131_(_env_, _kernel_result_274[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_274[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_274[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_274[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_467 = 37;
    }
        
        _result_[_tid_] = temp_stencil_467;
    }
}



// TODO: There should be a better to check if _block_k_133_ is already defined
#ifndef _block_k_133__func
#define _block_k_133__func
__device__ int _block_k_133_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_197 = ((({ int _temp_var_198 = ((({ int _temp_var_199 = ((values[2] % 4));
        (_temp_var_199 == 0 ? indices.field_0 : (_temp_var_199 == 1 ? indices.field_1 : (_temp_var_199 == 2 ? indices.field_2 : (_temp_var_199 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_198 == 0 ? indices.field_0 : (_temp_var_198 == 1 ? indices.field_1 : (_temp_var_198 == 2 ? indices.field_2 : (_temp_var_198 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_197 == 0 ? indices.field_0 : (_temp_var_197 == 1 ? indices.field_1 : (_temp_var_197 == 2 ? indices.field_2 : (_temp_var_197 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_269(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_272)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_468;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_468 = _block_k_133_(_env_, _kernel_result_272[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_272[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_272[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_272[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_468 = 37;
    }
        
        _result_[_tid_] = temp_stencil_468;
    }
}



// TODO: There should be a better to check if _block_k_135_ is already defined
#ifndef _block_k_135__func
#define _block_k_135__func
__device__ int _block_k_135_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_200 = ((({ int _temp_var_201 = ((({ int _temp_var_202 = ((values[2] % 4));
        (_temp_var_202 == 0 ? indices.field_0 : (_temp_var_202 == 1 ? indices.field_1 : (_temp_var_202 == 2 ? indices.field_2 : (_temp_var_202 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_201 == 0 ? indices.field_0 : (_temp_var_201 == 1 ? indices.field_1 : (_temp_var_201 == 2 ? indices.field_2 : (_temp_var_201 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_200 == 0 ? indices.field_0 : (_temp_var_200 == 1 ? indices.field_1 : (_temp_var_200 == 2 ? indices.field_2 : (_temp_var_200 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_267(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_270)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_469;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_469 = _block_k_135_(_env_, _kernel_result_270[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_270[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_270[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_270[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_469 = 37;
    }
        
        _result_[_tid_] = temp_stencil_469;
    }
}



// TODO: There should be a better to check if _block_k_137_ is already defined
#ifndef _block_k_137__func
#define _block_k_137__func
__device__ int _block_k_137_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_203 = ((({ int _temp_var_204 = ((({ int _temp_var_205 = ((values[2] % 4));
        (_temp_var_205 == 0 ? indices.field_0 : (_temp_var_205 == 1 ? indices.field_1 : (_temp_var_205 == 2 ? indices.field_2 : (_temp_var_205 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_204 == 0 ? indices.field_0 : (_temp_var_204 == 1 ? indices.field_1 : (_temp_var_204 == 2 ? indices.field_2 : (_temp_var_204 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_203 == 0 ? indices.field_0 : (_temp_var_203 == 1 ? indices.field_1 : (_temp_var_203 == 2 ? indices.field_2 : (_temp_var_203 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_265(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_268)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_470;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_470 = _block_k_137_(_env_, _kernel_result_268[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_268[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_268[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_268[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_470 = 37;
    }
        
        _result_[_tid_] = temp_stencil_470;
    }
}



// TODO: There should be a better to check if _block_k_139_ is already defined
#ifndef _block_k_139__func
#define _block_k_139__func
__device__ int _block_k_139_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_206 = ((({ int _temp_var_207 = ((({ int _temp_var_208 = ((values[2] % 4));
        (_temp_var_208 == 0 ? indices.field_0 : (_temp_var_208 == 1 ? indices.field_1 : (_temp_var_208 == 2 ? indices.field_2 : (_temp_var_208 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_207 == 0 ? indices.field_0 : (_temp_var_207 == 1 ? indices.field_1 : (_temp_var_207 == 2 ? indices.field_2 : (_temp_var_207 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_206 == 0 ? indices.field_0 : (_temp_var_206 == 1 ? indices.field_1 : (_temp_var_206 == 2 ? indices.field_2 : (_temp_var_206 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_263(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_266)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_471;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_471 = _block_k_139_(_env_, _kernel_result_266[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_266[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_266[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_266[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_471 = 37;
    }
        
        _result_[_tid_] = temp_stencil_471;
    }
}



// TODO: There should be a better to check if _block_k_141_ is already defined
#ifndef _block_k_141__func
#define _block_k_141__func
__device__ int _block_k_141_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_209 = ((({ int _temp_var_210 = ((({ int _temp_var_211 = ((values[2] % 4));
        (_temp_var_211 == 0 ? indices.field_0 : (_temp_var_211 == 1 ? indices.field_1 : (_temp_var_211 == 2 ? indices.field_2 : (_temp_var_211 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_210 == 0 ? indices.field_0 : (_temp_var_210 == 1 ? indices.field_1 : (_temp_var_210 == 2 ? indices.field_2 : (_temp_var_210 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_209 == 0 ? indices.field_0 : (_temp_var_209 == 1 ? indices.field_1 : (_temp_var_209 == 2 ? indices.field_2 : (_temp_var_209 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_261(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_264)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_472;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_472 = _block_k_141_(_env_, _kernel_result_264[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_264[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_264[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_264[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_472 = 37;
    }
        
        _result_[_tid_] = temp_stencil_472;
    }
}



// TODO: There should be a better to check if _block_k_143_ is already defined
#ifndef _block_k_143__func
#define _block_k_143__func
__device__ int _block_k_143_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_212 = ((({ int _temp_var_213 = ((({ int _temp_var_214 = ((values[2] % 4));
        (_temp_var_214 == 0 ? indices.field_0 : (_temp_var_214 == 1 ? indices.field_1 : (_temp_var_214 == 2 ? indices.field_2 : (_temp_var_214 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_213 == 0 ? indices.field_0 : (_temp_var_213 == 1 ? indices.field_1 : (_temp_var_213 == 2 ? indices.field_2 : (_temp_var_213 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_212 == 0 ? indices.field_0 : (_temp_var_212 == 1 ? indices.field_1 : (_temp_var_212 == 2 ? indices.field_2 : (_temp_var_212 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_259(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_262)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_473;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_473 = _block_k_143_(_env_, _kernel_result_262[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_262[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_262[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_262[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_473 = 37;
    }
        
        _result_[_tid_] = temp_stencil_473;
    }
}



// TODO: There should be a better to check if _block_k_145_ is already defined
#ifndef _block_k_145__func
#define _block_k_145__func
__device__ int _block_k_145_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_215 = ((({ int _temp_var_216 = ((({ int _temp_var_217 = ((values[2] % 4));
        (_temp_var_217 == 0 ? indices.field_0 : (_temp_var_217 == 1 ? indices.field_1 : (_temp_var_217 == 2 ? indices.field_2 : (_temp_var_217 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_216 == 0 ? indices.field_0 : (_temp_var_216 == 1 ? indices.field_1 : (_temp_var_216 == 2 ? indices.field_2 : (_temp_var_216 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_215 == 0 ? indices.field_0 : (_temp_var_215 == 1 ? indices.field_1 : (_temp_var_215 == 2 ? indices.field_2 : (_temp_var_215 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_257(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_260)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_474;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_474 = _block_k_145_(_env_, _kernel_result_260[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_260[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_260[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_260[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_474 = 37;
    }
        
        _result_[_tid_] = temp_stencil_474;
    }
}



// TODO: There should be a better to check if _block_k_147_ is already defined
#ifndef _block_k_147__func
#define _block_k_147__func
__device__ int _block_k_147_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_218 = ((({ int _temp_var_219 = ((({ int _temp_var_220 = ((values[2] % 4));
        (_temp_var_220 == 0 ? indices.field_0 : (_temp_var_220 == 1 ? indices.field_1 : (_temp_var_220 == 2 ? indices.field_2 : (_temp_var_220 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_219 == 0 ? indices.field_0 : (_temp_var_219 == 1 ? indices.field_1 : (_temp_var_219 == 2 ? indices.field_2 : (_temp_var_219 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_218 == 0 ? indices.field_0 : (_temp_var_218 == 1 ? indices.field_1 : (_temp_var_218 == 2 ? indices.field_2 : (_temp_var_218 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_255(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_258)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_475;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_475 = _block_k_147_(_env_, _kernel_result_258[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_258[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_258[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_258[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_475 = 37;
    }
        
        _result_[_tid_] = temp_stencil_475;
    }
}



// TODO: There should be a better to check if _block_k_149_ is already defined
#ifndef _block_k_149__func
#define _block_k_149__func
__device__ int _block_k_149_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_221 = ((({ int _temp_var_222 = ((({ int _temp_var_223 = ((values[2] % 4));
        (_temp_var_223 == 0 ? indices.field_0 : (_temp_var_223 == 1 ? indices.field_1 : (_temp_var_223 == 2 ? indices.field_2 : (_temp_var_223 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_222 == 0 ? indices.field_0 : (_temp_var_222 == 1 ? indices.field_1 : (_temp_var_222 == 2 ? indices.field_2 : (_temp_var_222 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_221 == 0 ? indices.field_0 : (_temp_var_221 == 1 ? indices.field_1 : (_temp_var_221 == 2 ? indices.field_2 : (_temp_var_221 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_253(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_256)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_476;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_476 = _block_k_149_(_env_, _kernel_result_256[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_256[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_256[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_256[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_476 = 37;
    }
        
        _result_[_tid_] = temp_stencil_476;
    }
}



// TODO: There should be a better to check if _block_k_151_ is already defined
#ifndef _block_k_151__func
#define _block_k_151__func
__device__ int _block_k_151_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_224 = ((({ int _temp_var_225 = ((({ int _temp_var_226 = ((values[2] % 4));
        (_temp_var_226 == 0 ? indices.field_0 : (_temp_var_226 == 1 ? indices.field_1 : (_temp_var_226 == 2 ? indices.field_2 : (_temp_var_226 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_225 == 0 ? indices.field_0 : (_temp_var_225 == 1 ? indices.field_1 : (_temp_var_225 == 2 ? indices.field_2 : (_temp_var_225 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_224 == 0 ? indices.field_0 : (_temp_var_224 == 1 ? indices.field_1 : (_temp_var_224 == 2 ? indices.field_2 : (_temp_var_224 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_251(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_254)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_477;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_477 = _block_k_151_(_env_, _kernel_result_254[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_254[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_254[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_254[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_477 = 37;
    }
        
        _result_[_tid_] = temp_stencil_477;
    }
}



// TODO: There should be a better to check if _block_k_153_ is already defined
#ifndef _block_k_153__func
#define _block_k_153__func
__device__ int _block_k_153_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_227 = ((({ int _temp_var_228 = ((({ int _temp_var_229 = ((values[2] % 4));
        (_temp_var_229 == 0 ? indices.field_0 : (_temp_var_229 == 1 ? indices.field_1 : (_temp_var_229 == 2 ? indices.field_2 : (_temp_var_229 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_228 == 0 ? indices.field_0 : (_temp_var_228 == 1 ? indices.field_1 : (_temp_var_228 == 2 ? indices.field_2 : (_temp_var_228 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_227 == 0 ? indices.field_0 : (_temp_var_227 == 1 ? indices.field_1 : (_temp_var_227 == 2 ? indices.field_2 : (_temp_var_227 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_249(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_252)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_478;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_478 = _block_k_153_(_env_, _kernel_result_252[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_252[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_252[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_252[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_478 = 37;
    }
        
        _result_[_tid_] = temp_stencil_478;
    }
}



// TODO: There should be a better to check if _block_k_155_ is already defined
#ifndef _block_k_155__func
#define _block_k_155__func
__device__ int _block_k_155_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_230 = ((({ int _temp_var_231 = ((({ int _temp_var_232 = ((values[2] % 4));
        (_temp_var_232 == 0 ? indices.field_0 : (_temp_var_232 == 1 ? indices.field_1 : (_temp_var_232 == 2 ? indices.field_2 : (_temp_var_232 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_231 == 0 ? indices.field_0 : (_temp_var_231 == 1 ? indices.field_1 : (_temp_var_231 == 2 ? indices.field_2 : (_temp_var_231 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_230 == 0 ? indices.field_0 : (_temp_var_230 == 1 ? indices.field_1 : (_temp_var_230 == 2 ? indices.field_2 : (_temp_var_230 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_247(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_250)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_479;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_479 = _block_k_155_(_env_, _kernel_result_250[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_250[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_250[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_250[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_479 = 37;
    }
        
        _result_[_tid_] = temp_stencil_479;
    }
}



// TODO: There should be a better to check if _block_k_157_ is already defined
#ifndef _block_k_157__func
#define _block_k_157__func
__device__ int _block_k_157_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_233 = ((({ int _temp_var_234 = ((({ int _temp_var_235 = ((values[2] % 4));
        (_temp_var_235 == 0 ? indices.field_0 : (_temp_var_235 == 1 ? indices.field_1 : (_temp_var_235 == 2 ? indices.field_2 : (_temp_var_235 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_234 == 0 ? indices.field_0 : (_temp_var_234 == 1 ? indices.field_1 : (_temp_var_234 == 2 ? indices.field_2 : (_temp_var_234 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_233 == 0 ? indices.field_0 : (_temp_var_233 == 1 ? indices.field_1 : (_temp_var_233 == 2 ? indices.field_2 : (_temp_var_233 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_245(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_248)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_480;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_480 = _block_k_157_(_env_, _kernel_result_248[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_248[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_248[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_248[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_480 = 37;
    }
        
        _result_[_tid_] = temp_stencil_480;
    }
}



// TODO: There should be a better to check if _block_k_159_ is already defined
#ifndef _block_k_159__func
#define _block_k_159__func
__device__ int _block_k_159_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_236 = ((({ int _temp_var_237 = ((({ int _temp_var_238 = ((values[2] % 4));
        (_temp_var_238 == 0 ? indices.field_0 : (_temp_var_238 == 1 ? indices.field_1 : (_temp_var_238 == 2 ? indices.field_2 : (_temp_var_238 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_237 == 0 ? indices.field_0 : (_temp_var_237 == 1 ? indices.field_1 : (_temp_var_237 == 2 ? indices.field_2 : (_temp_var_237 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_236 == 0 ? indices.field_0 : (_temp_var_236 == 1 ? indices.field_1 : (_temp_var_236 == 2 ? indices.field_2 : (_temp_var_236 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_243(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_246)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_481;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_481 = _block_k_159_(_env_, _kernel_result_246[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_246[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_246[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_246[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_481 = 37;
    }
        
        _result_[_tid_] = temp_stencil_481;
    }
}



// TODO: There should be a better to check if _block_k_161_ is already defined
#ifndef _block_k_161__func
#define _block_k_161__func
__device__ int _block_k_161_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_239 = ((({ int _temp_var_240 = ((({ int _temp_var_241 = ((values[2] % 4));
        (_temp_var_241 == 0 ? indices.field_0 : (_temp_var_241 == 1 ? indices.field_1 : (_temp_var_241 == 2 ? indices.field_2 : (_temp_var_241 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_240 == 0 ? indices.field_0 : (_temp_var_240 == 1 ? indices.field_1 : (_temp_var_240 == 2 ? indices.field_2 : (_temp_var_240 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_239 == 0 ? indices.field_0 : (_temp_var_239 == 1 ? indices.field_1 : (_temp_var_239 == 2 ? indices.field_2 : (_temp_var_239 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_241(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_244)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_482;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_482 = _block_k_161_(_env_, _kernel_result_244[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_244[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_244[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_244[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_482 = 37;
    }
        
        _result_[_tid_] = temp_stencil_482;
    }
}



// TODO: There should be a better to check if _block_k_163_ is already defined
#ifndef _block_k_163__func
#define _block_k_163__func
__device__ int _block_k_163_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_242 = ((({ int _temp_var_243 = ((({ int _temp_var_244 = ((values[2] % 4));
        (_temp_var_244 == 0 ? indices.field_0 : (_temp_var_244 == 1 ? indices.field_1 : (_temp_var_244 == 2 ? indices.field_2 : (_temp_var_244 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_243 == 0 ? indices.field_0 : (_temp_var_243 == 1 ? indices.field_1 : (_temp_var_243 == 2 ? indices.field_2 : (_temp_var_243 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_242 == 0 ? indices.field_0 : (_temp_var_242 == 1 ? indices.field_1 : (_temp_var_242 == 2 ? indices.field_2 : (_temp_var_242 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_239(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_242)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_483;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_483 = _block_k_163_(_env_, _kernel_result_242[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_242[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_242[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_242[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_483 = 37;
    }
        
        _result_[_tid_] = temp_stencil_483;
    }
}



// TODO: There should be a better to check if _block_k_165_ is already defined
#ifndef _block_k_165__func
#define _block_k_165__func
__device__ int _block_k_165_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_245 = ((({ int _temp_var_246 = ((({ int _temp_var_247 = ((values[2] % 4));
        (_temp_var_247 == 0 ? indices.field_0 : (_temp_var_247 == 1 ? indices.field_1 : (_temp_var_247 == 2 ? indices.field_2 : (_temp_var_247 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_246 == 0 ? indices.field_0 : (_temp_var_246 == 1 ? indices.field_1 : (_temp_var_246 == 2 ? indices.field_2 : (_temp_var_246 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_245 == 0 ? indices.field_0 : (_temp_var_245 == 1 ? indices.field_1 : (_temp_var_245 == 2 ? indices.field_2 : (_temp_var_245 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_237(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_240)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_484;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_484 = _block_k_165_(_env_, _kernel_result_240[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_240[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_240[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_240[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_484 = 37;
    }
        
        _result_[_tid_] = temp_stencil_484;
    }
}



// TODO: There should be a better to check if _block_k_167_ is already defined
#ifndef _block_k_167__func
#define _block_k_167__func
__device__ int _block_k_167_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_248 = ((({ int _temp_var_249 = ((({ int _temp_var_250 = ((values[2] % 4));
        (_temp_var_250 == 0 ? indices.field_0 : (_temp_var_250 == 1 ? indices.field_1 : (_temp_var_250 == 2 ? indices.field_2 : (_temp_var_250 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_249 == 0 ? indices.field_0 : (_temp_var_249 == 1 ? indices.field_1 : (_temp_var_249 == 2 ? indices.field_2 : (_temp_var_249 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_248 == 0 ? indices.field_0 : (_temp_var_248 == 1 ? indices.field_1 : (_temp_var_248 == 2 ? indices.field_2 : (_temp_var_248 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_235(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_238)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_485;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_485 = _block_k_167_(_env_, _kernel_result_238[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_238[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_238[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_238[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_485 = 37;
    }
        
        _result_[_tid_] = temp_stencil_485;
    }
}



// TODO: There should be a better to check if _block_k_169_ is already defined
#ifndef _block_k_169__func
#define _block_k_169__func
__device__ int _block_k_169_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_251 = ((({ int _temp_var_252 = ((({ int _temp_var_253 = ((values[2] % 4));
        (_temp_var_253 == 0 ? indices.field_0 : (_temp_var_253 == 1 ? indices.field_1 : (_temp_var_253 == 2 ? indices.field_2 : (_temp_var_253 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_252 == 0 ? indices.field_0 : (_temp_var_252 == 1 ? indices.field_1 : (_temp_var_252 == 2 ? indices.field_2 : (_temp_var_252 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_251 == 0 ? indices.field_0 : (_temp_var_251 == 1 ? indices.field_1 : (_temp_var_251 == 2 ? indices.field_2 : (_temp_var_251 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_233(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_236)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_486;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_486 = _block_k_169_(_env_, _kernel_result_236[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_236[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_236[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_236[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_486 = 37;
    }
        
        _result_[_tid_] = temp_stencil_486;
    }
}



// TODO: There should be a better to check if _block_k_171_ is already defined
#ifndef _block_k_171__func
#define _block_k_171__func
__device__ int _block_k_171_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_254 = ((({ int _temp_var_255 = ((({ int _temp_var_256 = ((values[2] % 4));
        (_temp_var_256 == 0 ? indices.field_0 : (_temp_var_256 == 1 ? indices.field_1 : (_temp_var_256 == 2 ? indices.field_2 : (_temp_var_256 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_255 == 0 ? indices.field_0 : (_temp_var_255 == 1 ? indices.field_1 : (_temp_var_255 == 2 ? indices.field_2 : (_temp_var_255 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_254 == 0 ? indices.field_0 : (_temp_var_254 == 1 ? indices.field_1 : (_temp_var_254 == 2 ? indices.field_2 : (_temp_var_254 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_231(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_234)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_487;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_487 = _block_k_171_(_env_, _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_487 = 37;
    }
        
        _result_[_tid_] = temp_stencil_487;
    }
}



// TODO: There should be a better to check if _block_k_173_ is already defined
#ifndef _block_k_173__func
#define _block_k_173__func
__device__ int _block_k_173_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_257 = ((({ int _temp_var_258 = ((({ int _temp_var_259 = ((values[2] % 4));
        (_temp_var_259 == 0 ? indices.field_0 : (_temp_var_259 == 1 ? indices.field_1 : (_temp_var_259 == 2 ? indices.field_2 : (_temp_var_259 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_258 == 0 ? indices.field_0 : (_temp_var_258 == 1 ? indices.field_1 : (_temp_var_258 == 2 ? indices.field_2 : (_temp_var_258 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_257 == 0 ? indices.field_0 : (_temp_var_257 == 1 ? indices.field_1 : (_temp_var_257 == 2 ? indices.field_2 : (_temp_var_257 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_229(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_232)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_488;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_488 = _block_k_173_(_env_, _kernel_result_232[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_232[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_232[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_232[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_488 = 37;
    }
        
        _result_[_tid_] = temp_stencil_488;
    }
}



// TODO: There should be a better to check if _block_k_175_ is already defined
#ifndef _block_k_175__func
#define _block_k_175__func
__device__ int _block_k_175_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_260 = ((({ int _temp_var_261 = ((({ int _temp_var_262 = ((values[2] % 4));
        (_temp_var_262 == 0 ? indices.field_0 : (_temp_var_262 == 1 ? indices.field_1 : (_temp_var_262 == 2 ? indices.field_2 : (_temp_var_262 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_261 == 0 ? indices.field_0 : (_temp_var_261 == 1 ? indices.field_1 : (_temp_var_261 == 2 ? indices.field_2 : (_temp_var_261 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_260 == 0 ? indices.field_0 : (_temp_var_260 == 1 ? indices.field_1 : (_temp_var_260 == 2 ? indices.field_2 : (_temp_var_260 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_227(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_230)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_489;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_489 = _block_k_175_(_env_, _kernel_result_230[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_230[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_230[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_230[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_489 = 37;
    }
        
        _result_[_tid_] = temp_stencil_489;
    }
}



// TODO: There should be a better to check if _block_k_177_ is already defined
#ifndef _block_k_177__func
#define _block_k_177__func
__device__ int _block_k_177_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_263 = ((({ int _temp_var_264 = ((({ int _temp_var_265 = ((values[2] % 4));
        (_temp_var_265 == 0 ? indices.field_0 : (_temp_var_265 == 1 ? indices.field_1 : (_temp_var_265 == 2 ? indices.field_2 : (_temp_var_265 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_264 == 0 ? indices.field_0 : (_temp_var_264 == 1 ? indices.field_1 : (_temp_var_264 == 2 ? indices.field_2 : (_temp_var_264 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_263 == 0 ? indices.field_0 : (_temp_var_263 == 1 ? indices.field_1 : (_temp_var_263 == 2 ? indices.field_2 : (_temp_var_263 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_225(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_228)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_490;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_490 = _block_k_177_(_env_, _kernel_result_228[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_228[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_228[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_228[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_490 = 37;
    }
        
        _result_[_tid_] = temp_stencil_490;
    }
}



// TODO: There should be a better to check if _block_k_179_ is already defined
#ifndef _block_k_179__func
#define _block_k_179__func
__device__ int _block_k_179_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_266 = ((({ int _temp_var_267 = ((({ int _temp_var_268 = ((values[2] % 4));
        (_temp_var_268 == 0 ? indices.field_0 : (_temp_var_268 == 1 ? indices.field_1 : (_temp_var_268 == 2 ? indices.field_2 : (_temp_var_268 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_267 == 0 ? indices.field_0 : (_temp_var_267 == 1 ? indices.field_1 : (_temp_var_267 == 2 ? indices.field_2 : (_temp_var_267 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_266 == 0 ? indices.field_0 : (_temp_var_266 == 1 ? indices.field_1 : (_temp_var_266 == 2 ? indices.field_2 : (_temp_var_266 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_223(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_226)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_491;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_491 = _block_k_179_(_env_, _kernel_result_226[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_226[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_226[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_226[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_491 = 37;
    }
        
        _result_[_tid_] = temp_stencil_491;
    }
}



// TODO: There should be a better to check if _block_k_181_ is already defined
#ifndef _block_k_181__func
#define _block_k_181__func
__device__ int _block_k_181_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_269 = ((({ int _temp_var_270 = ((({ int _temp_var_271 = ((values[2] % 4));
        (_temp_var_271 == 0 ? indices.field_0 : (_temp_var_271 == 1 ? indices.field_1 : (_temp_var_271 == 2 ? indices.field_2 : (_temp_var_271 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_270 == 0 ? indices.field_0 : (_temp_var_270 == 1 ? indices.field_1 : (_temp_var_270 == 2 ? indices.field_2 : (_temp_var_270 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_269 == 0 ? indices.field_0 : (_temp_var_269 == 1 ? indices.field_1 : (_temp_var_269 == 2 ? indices.field_2 : (_temp_var_269 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_221(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_224)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_492;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_492 = _block_k_181_(_env_, _kernel_result_224[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_224[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_224[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_224[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_492 = 37;
    }
        
        _result_[_tid_] = temp_stencil_492;
    }
}



// TODO: There should be a better to check if _block_k_183_ is already defined
#ifndef _block_k_183__func
#define _block_k_183__func
__device__ int _block_k_183_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_272 = ((({ int _temp_var_273 = ((({ int _temp_var_274 = ((values[2] % 4));
        (_temp_var_274 == 0 ? indices.field_0 : (_temp_var_274 == 1 ? indices.field_1 : (_temp_var_274 == 2 ? indices.field_2 : (_temp_var_274 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_273 == 0 ? indices.field_0 : (_temp_var_273 == 1 ? indices.field_1 : (_temp_var_273 == 2 ? indices.field_2 : (_temp_var_273 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_272 == 0 ? indices.field_0 : (_temp_var_272 == 1 ? indices.field_1 : (_temp_var_272 == 2 ? indices.field_2 : (_temp_var_272 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_219(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_222)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_493;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_493 = _block_k_183_(_env_, _kernel_result_222[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_222[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_222[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_222[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_493 = 37;
    }
        
        _result_[_tid_] = temp_stencil_493;
    }
}



// TODO: There should be a better to check if _block_k_185_ is already defined
#ifndef _block_k_185__func
#define _block_k_185__func
__device__ int _block_k_185_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_275 = ((({ int _temp_var_276 = ((({ int _temp_var_277 = ((values[2] % 4));
        (_temp_var_277 == 0 ? indices.field_0 : (_temp_var_277 == 1 ? indices.field_1 : (_temp_var_277 == 2 ? indices.field_2 : (_temp_var_277 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_276 == 0 ? indices.field_0 : (_temp_var_276 == 1 ? indices.field_1 : (_temp_var_276 == 2 ? indices.field_2 : (_temp_var_276 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_275 == 0 ? indices.field_0 : (_temp_var_275 == 1 ? indices.field_1 : (_temp_var_275 == 2 ? indices.field_2 : (_temp_var_275 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_217(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_220)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_494;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_494 = _block_k_185_(_env_, _kernel_result_220[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_220[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_220[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_220[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_494 = 37;
    }
        
        _result_[_tid_] = temp_stencil_494;
    }
}



// TODO: There should be a better to check if _block_k_187_ is already defined
#ifndef _block_k_187__func
#define _block_k_187__func
__device__ int _block_k_187_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_278 = ((({ int _temp_var_279 = ((({ int _temp_var_280 = ((values[2] % 4));
        (_temp_var_280 == 0 ? indices.field_0 : (_temp_var_280 == 1 ? indices.field_1 : (_temp_var_280 == 2 ? indices.field_2 : (_temp_var_280 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_279 == 0 ? indices.field_0 : (_temp_var_279 == 1 ? indices.field_1 : (_temp_var_279 == 2 ? indices.field_2 : (_temp_var_279 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_278 == 0 ? indices.field_0 : (_temp_var_278 == 1 ? indices.field_1 : (_temp_var_278 == 2 ? indices.field_2 : (_temp_var_278 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_215(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_218)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_495;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_495 = _block_k_187_(_env_, _kernel_result_218[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_218[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_218[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_218[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_495 = 37;
    }
        
        _result_[_tid_] = temp_stencil_495;
    }
}



// TODO: There should be a better to check if _block_k_189_ is already defined
#ifndef _block_k_189__func
#define _block_k_189__func
__device__ int _block_k_189_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_281 = ((({ int _temp_var_282 = ((({ int _temp_var_283 = ((values[2] % 4));
        (_temp_var_283 == 0 ? indices.field_0 : (_temp_var_283 == 1 ? indices.field_1 : (_temp_var_283 == 2 ? indices.field_2 : (_temp_var_283 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_282 == 0 ? indices.field_0 : (_temp_var_282 == 1 ? indices.field_1 : (_temp_var_282 == 2 ? indices.field_2 : (_temp_var_282 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_281 == 0 ? indices.field_0 : (_temp_var_281 == 1 ? indices.field_1 : (_temp_var_281 == 2 ? indices.field_2 : (_temp_var_281 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_213(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_216)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_496;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_496 = _block_k_189_(_env_, _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_496 = 37;
    }
        
        _result_[_tid_] = temp_stencil_496;
    }
}



// TODO: There should be a better to check if _block_k_191_ is already defined
#ifndef _block_k_191__func
#define _block_k_191__func
__device__ int _block_k_191_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_284 = ((({ int _temp_var_285 = ((({ int _temp_var_286 = ((values[2] % 4));
        (_temp_var_286 == 0 ? indices.field_0 : (_temp_var_286 == 1 ? indices.field_1 : (_temp_var_286 == 2 ? indices.field_2 : (_temp_var_286 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_285 == 0 ? indices.field_0 : (_temp_var_285 == 1 ? indices.field_1 : (_temp_var_285 == 2 ? indices.field_2 : (_temp_var_285 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_284 == 0 ? indices.field_0 : (_temp_var_284 == 1 ? indices.field_1 : (_temp_var_284 == 2 ? indices.field_2 : (_temp_var_284 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_211(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_214)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_497;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_497 = _block_k_191_(_env_, _kernel_result_214[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_214[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_214[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_214[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_497 = 37;
    }
        
        _result_[_tid_] = temp_stencil_497;
    }
}



// TODO: There should be a better to check if _block_k_193_ is already defined
#ifndef _block_k_193__func
#define _block_k_193__func
__device__ int _block_k_193_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_287 = ((({ int _temp_var_288 = ((({ int _temp_var_289 = ((values[2] % 4));
        (_temp_var_289 == 0 ? indices.field_0 : (_temp_var_289 == 1 ? indices.field_1 : (_temp_var_289 == 2 ? indices.field_2 : (_temp_var_289 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_288 == 0 ? indices.field_0 : (_temp_var_288 == 1 ? indices.field_1 : (_temp_var_288 == 2 ? indices.field_2 : (_temp_var_288 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_287 == 0 ? indices.field_0 : (_temp_var_287 == 1 ? indices.field_1 : (_temp_var_287 == 2 ? indices.field_2 : (_temp_var_287 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_209(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_212)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_498;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_498 = _block_k_193_(_env_, _kernel_result_212[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_212[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_212[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_212[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_498 = 37;
    }
        
        _result_[_tid_] = temp_stencil_498;
    }
}



// TODO: There should be a better to check if _block_k_195_ is already defined
#ifndef _block_k_195__func
#define _block_k_195__func
__device__ int _block_k_195_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_290 = ((({ int _temp_var_291 = ((({ int _temp_var_292 = ((values[2] % 4));
        (_temp_var_292 == 0 ? indices.field_0 : (_temp_var_292 == 1 ? indices.field_1 : (_temp_var_292 == 2 ? indices.field_2 : (_temp_var_292 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_291 == 0 ? indices.field_0 : (_temp_var_291 == 1 ? indices.field_1 : (_temp_var_291 == 2 ? indices.field_2 : (_temp_var_291 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_290 == 0 ? indices.field_0 : (_temp_var_290 == 1 ? indices.field_1 : (_temp_var_290 == 2 ? indices.field_2 : (_temp_var_290 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_207(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_210)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_499;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_499 = _block_k_195_(_env_, _kernel_result_210[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_210[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_210[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_210[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_499 = 37;
    }
        
        _result_[_tid_] = temp_stencil_499;
    }
}



// TODO: There should be a better to check if _block_k_197_ is already defined
#ifndef _block_k_197__func
#define _block_k_197__func
__device__ int _block_k_197_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_293 = ((({ int _temp_var_294 = ((({ int _temp_var_295 = ((values[2] % 4));
        (_temp_var_295 == 0 ? indices.field_0 : (_temp_var_295 == 1 ? indices.field_1 : (_temp_var_295 == 2 ? indices.field_2 : (_temp_var_295 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_294 == 0 ? indices.field_0 : (_temp_var_294 == 1 ? indices.field_1 : (_temp_var_294 == 2 ? indices.field_2 : (_temp_var_294 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_293 == 0 ? indices.field_0 : (_temp_var_293 == 1 ? indices.field_1 : (_temp_var_293 == 2 ? indices.field_2 : (_temp_var_293 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_205(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_208)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_500;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_500 = _block_k_197_(_env_, _kernel_result_208[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_208[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_208[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_208[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_500 = 37;
    }
        
        _result_[_tid_] = temp_stencil_500;
    }
}



// TODO: There should be a better to check if _block_k_199_ is already defined
#ifndef _block_k_199__func
#define _block_k_199__func
__device__ int _block_k_199_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_296 = ((({ int _temp_var_297 = ((({ int _temp_var_298 = ((values[2] % 4));
        (_temp_var_298 == 0 ? indices.field_0 : (_temp_var_298 == 1 ? indices.field_1 : (_temp_var_298 == 2 ? indices.field_2 : (_temp_var_298 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_297 == 0 ? indices.field_0 : (_temp_var_297 == 1 ? indices.field_1 : (_temp_var_297 == 2 ? indices.field_2 : (_temp_var_297 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_296 == 0 ? indices.field_0 : (_temp_var_296 == 1 ? indices.field_1 : (_temp_var_296 == 2 ? indices.field_2 : (_temp_var_296 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_203(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_206)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_501;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_501 = _block_k_199_(_env_, _kernel_result_206[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_206[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_206[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_206[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_501 = 37;
    }
        
        _result_[_tid_] = temp_stencil_501;
    }
}



// TODO: There should be a better to check if _block_k_201_ is already defined
#ifndef _block_k_201__func
#define _block_k_201__func
__device__ int _block_k_201_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_299 = ((({ int _temp_var_300 = ((({ int _temp_var_301 = ((values[2] % 4));
        (_temp_var_301 == 0 ? indices.field_0 : (_temp_var_301 == 1 ? indices.field_1 : (_temp_var_301 == 2 ? indices.field_2 : (_temp_var_301 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_300 == 0 ? indices.field_0 : (_temp_var_300 == 1 ? indices.field_1 : (_temp_var_300 == 2 ? indices.field_2 : (_temp_var_300 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_299 == 0 ? indices.field_0 : (_temp_var_299 == 1 ? indices.field_1 : (_temp_var_299 == 2 ? indices.field_2 : (_temp_var_299 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_201(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_204)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_502;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_502 = _block_k_201_(_env_, _kernel_result_204[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_204[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_204[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_204[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_502 = 37;
    }
        
        _result_[_tid_] = temp_stencil_502;
    }
}



// TODO: There should be a better to check if _block_k_203_ is already defined
#ifndef _block_k_203__func
#define _block_k_203__func
__device__ int _block_k_203_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_302 = ((({ int _temp_var_303 = ((({ int _temp_var_304 = ((values[2] % 4));
        (_temp_var_304 == 0 ? indices.field_0 : (_temp_var_304 == 1 ? indices.field_1 : (_temp_var_304 == 2 ? indices.field_2 : (_temp_var_304 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_303 == 0 ? indices.field_0 : (_temp_var_303 == 1 ? indices.field_1 : (_temp_var_303 == 2 ? indices.field_2 : (_temp_var_303 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_302 == 0 ? indices.field_0 : (_temp_var_302 == 1 ? indices.field_1 : (_temp_var_302 == 2 ? indices.field_2 : (_temp_var_302 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_199(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_202)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_503;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_503 = _block_k_203_(_env_, _kernel_result_202[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_202[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_202[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_202[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_503 = 37;
    }
        
        _result_[_tid_] = temp_stencil_503;
    }
}



// TODO: There should be a better to check if _block_k_205_ is already defined
#ifndef _block_k_205__func
#define _block_k_205__func
__device__ int _block_k_205_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_305 = ((({ int _temp_var_306 = ((({ int _temp_var_307 = ((values[2] % 4));
        (_temp_var_307 == 0 ? indices.field_0 : (_temp_var_307 == 1 ? indices.field_1 : (_temp_var_307 == 2 ? indices.field_2 : (_temp_var_307 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_306 == 0 ? indices.field_0 : (_temp_var_306 == 1 ? indices.field_1 : (_temp_var_306 == 2 ? indices.field_2 : (_temp_var_306 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_305 == 0 ? indices.field_0 : (_temp_var_305 == 1 ? indices.field_1 : (_temp_var_305 == 2 ? indices.field_2 : (_temp_var_305 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_197(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_200)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_504;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_504 = _block_k_205_(_env_, _kernel_result_200[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_200[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_200[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_200[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_504 = 37;
    }
        
        _result_[_tid_] = temp_stencil_504;
    }
}



// TODO: There should be a better to check if _block_k_207_ is already defined
#ifndef _block_k_207__func
#define _block_k_207__func
__device__ int _block_k_207_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_308 = ((({ int _temp_var_309 = ((({ int _temp_var_310 = ((values[2] % 4));
        (_temp_var_310 == 0 ? indices.field_0 : (_temp_var_310 == 1 ? indices.field_1 : (_temp_var_310 == 2 ? indices.field_2 : (_temp_var_310 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_309 == 0 ? indices.field_0 : (_temp_var_309 == 1 ? indices.field_1 : (_temp_var_309 == 2 ? indices.field_2 : (_temp_var_309 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_308 == 0 ? indices.field_0 : (_temp_var_308 == 1 ? indices.field_1 : (_temp_var_308 == 2 ? indices.field_2 : (_temp_var_308 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_195(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_198)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_505;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_505 = _block_k_207_(_env_, _kernel_result_198[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_198[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_198[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_198[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_505 = 37;
    }
        
        _result_[_tid_] = temp_stencil_505;
    }
}



// TODO: There should be a better to check if _block_k_209_ is already defined
#ifndef _block_k_209__func
#define _block_k_209__func
__device__ int _block_k_209_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_311 = ((({ int _temp_var_312 = ((({ int _temp_var_313 = ((values[2] % 4));
        (_temp_var_313 == 0 ? indices.field_0 : (_temp_var_313 == 1 ? indices.field_1 : (_temp_var_313 == 2 ? indices.field_2 : (_temp_var_313 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_312 == 0 ? indices.field_0 : (_temp_var_312 == 1 ? indices.field_1 : (_temp_var_312 == 2 ? indices.field_2 : (_temp_var_312 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_311 == 0 ? indices.field_0 : (_temp_var_311 == 1 ? indices.field_1 : (_temp_var_311 == 2 ? indices.field_2 : (_temp_var_311 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_193(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_196)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_506;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_506 = _block_k_209_(_env_, _kernel_result_196[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_196[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_196[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_196[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_506 = 37;
    }
        
        _result_[_tid_] = temp_stencil_506;
    }
}



// TODO: There should be a better to check if _block_k_211_ is already defined
#ifndef _block_k_211__func
#define _block_k_211__func
__device__ int _block_k_211_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_314 = ((({ int _temp_var_315 = ((({ int _temp_var_316 = ((values[2] % 4));
        (_temp_var_316 == 0 ? indices.field_0 : (_temp_var_316 == 1 ? indices.field_1 : (_temp_var_316 == 2 ? indices.field_2 : (_temp_var_316 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_315 == 0 ? indices.field_0 : (_temp_var_315 == 1 ? indices.field_1 : (_temp_var_315 == 2 ? indices.field_2 : (_temp_var_315 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_314 == 0 ? indices.field_0 : (_temp_var_314 == 1 ? indices.field_1 : (_temp_var_314 == 2 ? indices.field_2 : (_temp_var_314 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_191(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_194)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_507;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_507 = _block_k_211_(_env_, _kernel_result_194[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_194[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_194[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_194[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_507 = 37;
    }
        
        _result_[_tid_] = temp_stencil_507;
    }
}



// TODO: There should be a better to check if _block_k_213_ is already defined
#ifndef _block_k_213__func
#define _block_k_213__func
__device__ int _block_k_213_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_317 = ((({ int _temp_var_318 = ((({ int _temp_var_319 = ((values[2] % 4));
        (_temp_var_319 == 0 ? indices.field_0 : (_temp_var_319 == 1 ? indices.field_1 : (_temp_var_319 == 2 ? indices.field_2 : (_temp_var_319 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_318 == 0 ? indices.field_0 : (_temp_var_318 == 1 ? indices.field_1 : (_temp_var_318 == 2 ? indices.field_2 : (_temp_var_318 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_317 == 0 ? indices.field_0 : (_temp_var_317 == 1 ? indices.field_1 : (_temp_var_317 == 2 ? indices.field_2 : (_temp_var_317 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_189(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_192)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_508;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_508 = _block_k_213_(_env_, _kernel_result_192[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_192[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_192[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_192[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_508 = 37;
    }
        
        _result_[_tid_] = temp_stencil_508;
    }
}



// TODO: There should be a better to check if _block_k_215_ is already defined
#ifndef _block_k_215__func
#define _block_k_215__func
__device__ int _block_k_215_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_320 = ((({ int _temp_var_321 = ((({ int _temp_var_322 = ((values[2] % 4));
        (_temp_var_322 == 0 ? indices.field_0 : (_temp_var_322 == 1 ? indices.field_1 : (_temp_var_322 == 2 ? indices.field_2 : (_temp_var_322 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_321 == 0 ? indices.field_0 : (_temp_var_321 == 1 ? indices.field_1 : (_temp_var_321 == 2 ? indices.field_2 : (_temp_var_321 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_320 == 0 ? indices.field_0 : (_temp_var_320 == 1 ? indices.field_1 : (_temp_var_320 == 2 ? indices.field_2 : (_temp_var_320 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_187(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_190)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_509;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_509 = _block_k_215_(_env_, _kernel_result_190[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_190[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_190[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_190[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_509 = 37;
    }
        
        _result_[_tid_] = temp_stencil_509;
    }
}



// TODO: There should be a better to check if _block_k_217_ is already defined
#ifndef _block_k_217__func
#define _block_k_217__func
__device__ int _block_k_217_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_323 = ((({ int _temp_var_324 = ((({ int _temp_var_325 = ((values[2] % 4));
        (_temp_var_325 == 0 ? indices.field_0 : (_temp_var_325 == 1 ? indices.field_1 : (_temp_var_325 == 2 ? indices.field_2 : (_temp_var_325 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_324 == 0 ? indices.field_0 : (_temp_var_324 == 1 ? indices.field_1 : (_temp_var_324 == 2 ? indices.field_2 : (_temp_var_324 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_323 == 0 ? indices.field_0 : (_temp_var_323 == 1 ? indices.field_1 : (_temp_var_323 == 2 ? indices.field_2 : (_temp_var_323 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_185(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_188)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_510;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_510 = _block_k_217_(_env_, _kernel_result_188[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_188[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_188[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_188[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_510 = 37;
    }
        
        _result_[_tid_] = temp_stencil_510;
    }
}



// TODO: There should be a better to check if _block_k_219_ is already defined
#ifndef _block_k_219__func
#define _block_k_219__func
__device__ int _block_k_219_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_326 = ((({ int _temp_var_327 = ((({ int _temp_var_328 = ((values[2] % 4));
        (_temp_var_328 == 0 ? indices.field_0 : (_temp_var_328 == 1 ? indices.field_1 : (_temp_var_328 == 2 ? indices.field_2 : (_temp_var_328 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_327 == 0 ? indices.field_0 : (_temp_var_327 == 1 ? indices.field_1 : (_temp_var_327 == 2 ? indices.field_2 : (_temp_var_327 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_326 == 0 ? indices.field_0 : (_temp_var_326 == 1 ? indices.field_1 : (_temp_var_326 == 2 ? indices.field_2 : (_temp_var_326 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_183(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_186)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_511;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_511 = _block_k_219_(_env_, _kernel_result_186[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_186[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_186[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_186[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_511 = 37;
    }
        
        _result_[_tid_] = temp_stencil_511;
    }
}



// TODO: There should be a better to check if _block_k_221_ is already defined
#ifndef _block_k_221__func
#define _block_k_221__func
__device__ int _block_k_221_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_329 = ((({ int _temp_var_330 = ((({ int _temp_var_331 = ((values[2] % 4));
        (_temp_var_331 == 0 ? indices.field_0 : (_temp_var_331 == 1 ? indices.field_1 : (_temp_var_331 == 2 ? indices.field_2 : (_temp_var_331 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_330 == 0 ? indices.field_0 : (_temp_var_330 == 1 ? indices.field_1 : (_temp_var_330 == 2 ? indices.field_2 : (_temp_var_330 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_329 == 0 ? indices.field_0 : (_temp_var_329 == 1 ? indices.field_1 : (_temp_var_329 == 2 ? indices.field_2 : (_temp_var_329 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_181(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_184)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_512;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_512 = _block_k_221_(_env_, _kernel_result_184[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_184[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_184[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_184[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_512 = 37;
    }
        
        _result_[_tid_] = temp_stencil_512;
    }
}



// TODO: There should be a better to check if _block_k_223_ is already defined
#ifndef _block_k_223__func
#define _block_k_223__func
__device__ int _block_k_223_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_332 = ((({ int _temp_var_333 = ((({ int _temp_var_334 = ((values[2] % 4));
        (_temp_var_334 == 0 ? indices.field_0 : (_temp_var_334 == 1 ? indices.field_1 : (_temp_var_334 == 2 ? indices.field_2 : (_temp_var_334 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_333 == 0 ? indices.field_0 : (_temp_var_333 == 1 ? indices.field_1 : (_temp_var_333 == 2 ? indices.field_2 : (_temp_var_333 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_332 == 0 ? indices.field_0 : (_temp_var_332 == 1 ? indices.field_1 : (_temp_var_332 == 2 ? indices.field_2 : (_temp_var_332 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_179(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_182)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_513;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_513 = _block_k_223_(_env_, _kernel_result_182[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_182[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_182[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_182[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_513 = 37;
    }
        
        _result_[_tid_] = temp_stencil_513;
    }
}



// TODO: There should be a better to check if _block_k_225_ is already defined
#ifndef _block_k_225__func
#define _block_k_225__func
__device__ int _block_k_225_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_335 = ((({ int _temp_var_336 = ((({ int _temp_var_337 = ((values[2] % 4));
        (_temp_var_337 == 0 ? indices.field_0 : (_temp_var_337 == 1 ? indices.field_1 : (_temp_var_337 == 2 ? indices.field_2 : (_temp_var_337 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_336 == 0 ? indices.field_0 : (_temp_var_336 == 1 ? indices.field_1 : (_temp_var_336 == 2 ? indices.field_2 : (_temp_var_336 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_335 == 0 ? indices.field_0 : (_temp_var_335 == 1 ? indices.field_1 : (_temp_var_335 == 2 ? indices.field_2 : (_temp_var_335 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_177(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_180)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_514;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_514 = _block_k_225_(_env_, _kernel_result_180[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_180[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_180[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_180[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_514 = 37;
    }
        
        _result_[_tid_] = temp_stencil_514;
    }
}



// TODO: There should be a better to check if _block_k_227_ is already defined
#ifndef _block_k_227__func
#define _block_k_227__func
__device__ int _block_k_227_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_338 = ((({ int _temp_var_339 = ((({ int _temp_var_340 = ((values[2] % 4));
        (_temp_var_340 == 0 ? indices.field_0 : (_temp_var_340 == 1 ? indices.field_1 : (_temp_var_340 == 2 ? indices.field_2 : (_temp_var_340 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_339 == 0 ? indices.field_0 : (_temp_var_339 == 1 ? indices.field_1 : (_temp_var_339 == 2 ? indices.field_2 : (_temp_var_339 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_338 == 0 ? indices.field_0 : (_temp_var_338 == 1 ? indices.field_1 : (_temp_var_338 == 2 ? indices.field_2 : (_temp_var_338 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_175(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_178)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_515;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_515 = _block_k_227_(_env_, _kernel_result_178[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_178[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_178[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_178[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_515 = 37;
    }
        
        _result_[_tid_] = temp_stencil_515;
    }
}



// TODO: There should be a better to check if _block_k_229_ is already defined
#ifndef _block_k_229__func
#define _block_k_229__func
__device__ int _block_k_229_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_341 = ((({ int _temp_var_342 = ((({ int _temp_var_343 = ((values[2] % 4));
        (_temp_var_343 == 0 ? indices.field_0 : (_temp_var_343 == 1 ? indices.field_1 : (_temp_var_343 == 2 ? indices.field_2 : (_temp_var_343 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_342 == 0 ? indices.field_0 : (_temp_var_342 == 1 ? indices.field_1 : (_temp_var_342 == 2 ? indices.field_2 : (_temp_var_342 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_341 == 0 ? indices.field_0 : (_temp_var_341 == 1 ? indices.field_1 : (_temp_var_341 == 2 ? indices.field_2 : (_temp_var_341 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_173(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_176)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_516;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_516 = _block_k_229_(_env_, _kernel_result_176[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_176[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_176[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_176[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_516 = 37;
    }
        
        _result_[_tid_] = temp_stencil_516;
    }
}



// TODO: There should be a better to check if _block_k_231_ is already defined
#ifndef _block_k_231__func
#define _block_k_231__func
__device__ int _block_k_231_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_344 = ((({ int _temp_var_345 = ((({ int _temp_var_346 = ((values[2] % 4));
        (_temp_var_346 == 0 ? indices.field_0 : (_temp_var_346 == 1 ? indices.field_1 : (_temp_var_346 == 2 ? indices.field_2 : (_temp_var_346 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_345 == 0 ? indices.field_0 : (_temp_var_345 == 1 ? indices.field_1 : (_temp_var_345 == 2 ? indices.field_2 : (_temp_var_345 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_344 == 0 ? indices.field_0 : (_temp_var_344 == 1 ? indices.field_1 : (_temp_var_344 == 2 ? indices.field_2 : (_temp_var_344 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_171(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_174)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_517;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_517 = _block_k_231_(_env_, _kernel_result_174[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_174[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_174[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_174[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_517 = 37;
    }
        
        _result_[_tid_] = temp_stencil_517;
    }
}



// TODO: There should be a better to check if _block_k_233_ is already defined
#ifndef _block_k_233__func
#define _block_k_233__func
__device__ int _block_k_233_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_347 = ((({ int _temp_var_348 = ((({ int _temp_var_349 = ((values[2] % 4));
        (_temp_var_349 == 0 ? indices.field_0 : (_temp_var_349 == 1 ? indices.field_1 : (_temp_var_349 == 2 ? indices.field_2 : (_temp_var_349 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_348 == 0 ? indices.field_0 : (_temp_var_348 == 1 ? indices.field_1 : (_temp_var_348 == 2 ? indices.field_2 : (_temp_var_348 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_347 == 0 ? indices.field_0 : (_temp_var_347 == 1 ? indices.field_1 : (_temp_var_347 == 2 ? indices.field_2 : (_temp_var_347 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_169(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_172)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_518;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_518 = _block_k_233_(_env_, _kernel_result_172[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_172[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_172[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_172[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_518 = 37;
    }
        
        _result_[_tid_] = temp_stencil_518;
    }
}



// TODO: There should be a better to check if _block_k_235_ is already defined
#ifndef _block_k_235__func
#define _block_k_235__func
__device__ int _block_k_235_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_350 = ((({ int _temp_var_351 = ((({ int _temp_var_352 = ((values[2] % 4));
        (_temp_var_352 == 0 ? indices.field_0 : (_temp_var_352 == 1 ? indices.field_1 : (_temp_var_352 == 2 ? indices.field_2 : (_temp_var_352 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_351 == 0 ? indices.field_0 : (_temp_var_351 == 1 ? indices.field_1 : (_temp_var_351 == 2 ? indices.field_2 : (_temp_var_351 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_350 == 0 ? indices.field_0 : (_temp_var_350 == 1 ? indices.field_1 : (_temp_var_350 == 2 ? indices.field_2 : (_temp_var_350 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_167(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_170)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_519;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_519 = _block_k_235_(_env_, _kernel_result_170[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_170[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_170[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_170[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_519 = 37;
    }
        
        _result_[_tid_] = temp_stencil_519;
    }
}



// TODO: There should be a better to check if _block_k_237_ is already defined
#ifndef _block_k_237__func
#define _block_k_237__func
__device__ int _block_k_237_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_353 = ((({ int _temp_var_354 = ((({ int _temp_var_355 = ((values[2] % 4));
        (_temp_var_355 == 0 ? indices.field_0 : (_temp_var_355 == 1 ? indices.field_1 : (_temp_var_355 == 2 ? indices.field_2 : (_temp_var_355 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_354 == 0 ? indices.field_0 : (_temp_var_354 == 1 ? indices.field_1 : (_temp_var_354 == 2 ? indices.field_2 : (_temp_var_354 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_353 == 0 ? indices.field_0 : (_temp_var_353 == 1 ? indices.field_1 : (_temp_var_353 == 2 ? indices.field_2 : (_temp_var_353 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_165(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_168)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_520;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_520 = _block_k_237_(_env_, _kernel_result_168[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_168[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_168[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_168[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_520 = 37;
    }
        
        _result_[_tid_] = temp_stencil_520;
    }
}



// TODO: There should be a better to check if _block_k_239_ is already defined
#ifndef _block_k_239__func
#define _block_k_239__func
__device__ int _block_k_239_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_356 = ((({ int _temp_var_357 = ((({ int _temp_var_358 = ((values[2] % 4));
        (_temp_var_358 == 0 ? indices.field_0 : (_temp_var_358 == 1 ? indices.field_1 : (_temp_var_358 == 2 ? indices.field_2 : (_temp_var_358 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_357 == 0 ? indices.field_0 : (_temp_var_357 == 1 ? indices.field_1 : (_temp_var_357 == 2 ? indices.field_2 : (_temp_var_357 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_356 == 0 ? indices.field_0 : (_temp_var_356 == 1 ? indices.field_1 : (_temp_var_356 == 2 ? indices.field_2 : (_temp_var_356 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_163(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_166)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_521;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_521 = _block_k_239_(_env_, _kernel_result_166[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_166[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_166[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_166[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_521 = 37;
    }
        
        _result_[_tid_] = temp_stencil_521;
    }
}



// TODO: There should be a better to check if _block_k_241_ is already defined
#ifndef _block_k_241__func
#define _block_k_241__func
__device__ int _block_k_241_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_359 = ((({ int _temp_var_360 = ((({ int _temp_var_361 = ((values[2] % 4));
        (_temp_var_361 == 0 ? indices.field_0 : (_temp_var_361 == 1 ? indices.field_1 : (_temp_var_361 == 2 ? indices.field_2 : (_temp_var_361 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_360 == 0 ? indices.field_0 : (_temp_var_360 == 1 ? indices.field_1 : (_temp_var_360 == 2 ? indices.field_2 : (_temp_var_360 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_359 == 0 ? indices.field_0 : (_temp_var_359 == 1 ? indices.field_1 : (_temp_var_359 == 2 ? indices.field_2 : (_temp_var_359 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_161(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_164)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_522;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_522 = _block_k_241_(_env_, _kernel_result_164[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_164[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_164[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_164[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_522 = 37;
    }
        
        _result_[_tid_] = temp_stencil_522;
    }
}



// TODO: There should be a better to check if _block_k_243_ is already defined
#ifndef _block_k_243__func
#define _block_k_243__func
__device__ int _block_k_243_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_362 = ((({ int _temp_var_363 = ((({ int _temp_var_364 = ((values[2] % 4));
        (_temp_var_364 == 0 ? indices.field_0 : (_temp_var_364 == 1 ? indices.field_1 : (_temp_var_364 == 2 ? indices.field_2 : (_temp_var_364 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_363 == 0 ? indices.field_0 : (_temp_var_363 == 1 ? indices.field_1 : (_temp_var_363 == 2 ? indices.field_2 : (_temp_var_363 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_362 == 0 ? indices.field_0 : (_temp_var_362 == 1 ? indices.field_1 : (_temp_var_362 == 2 ? indices.field_2 : (_temp_var_362 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_159(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_162)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_523;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_523 = _block_k_243_(_env_, _kernel_result_162[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_162[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_162[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_162[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_523 = 37;
    }
        
        _result_[_tid_] = temp_stencil_523;
    }
}



// TODO: There should be a better to check if _block_k_245_ is already defined
#ifndef _block_k_245__func
#define _block_k_245__func
__device__ int _block_k_245_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_365 = ((({ int _temp_var_366 = ((({ int _temp_var_367 = ((values[2] % 4));
        (_temp_var_367 == 0 ? indices.field_0 : (_temp_var_367 == 1 ? indices.field_1 : (_temp_var_367 == 2 ? indices.field_2 : (_temp_var_367 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_366 == 0 ? indices.field_0 : (_temp_var_366 == 1 ? indices.field_1 : (_temp_var_366 == 2 ? indices.field_2 : (_temp_var_366 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_365 == 0 ? indices.field_0 : (_temp_var_365 == 1 ? indices.field_1 : (_temp_var_365 == 2 ? indices.field_2 : (_temp_var_365 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_157(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_160)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_524;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_524 = _block_k_245_(_env_, _kernel_result_160[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_160[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_160[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_160[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_524 = 37;
    }
        
        _result_[_tid_] = temp_stencil_524;
    }
}



// TODO: There should be a better to check if _block_k_247_ is already defined
#ifndef _block_k_247__func
#define _block_k_247__func
__device__ int _block_k_247_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_368 = ((({ int _temp_var_369 = ((({ int _temp_var_370 = ((values[2] % 4));
        (_temp_var_370 == 0 ? indices.field_0 : (_temp_var_370 == 1 ? indices.field_1 : (_temp_var_370 == 2 ? indices.field_2 : (_temp_var_370 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_369 == 0 ? indices.field_0 : (_temp_var_369 == 1 ? indices.field_1 : (_temp_var_369 == 2 ? indices.field_2 : (_temp_var_369 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_368 == 0 ? indices.field_0 : (_temp_var_368 == 1 ? indices.field_1 : (_temp_var_368 == 2 ? indices.field_2 : (_temp_var_368 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_155(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_158)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_525;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_525 = _block_k_247_(_env_, _kernel_result_158[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_158[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_158[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_158[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_525 = 37;
    }
        
        _result_[_tid_] = temp_stencil_525;
    }
}



// TODO: There should be a better to check if _block_k_249_ is already defined
#ifndef _block_k_249__func
#define _block_k_249__func
__device__ int _block_k_249_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_371 = ((({ int _temp_var_372 = ((({ int _temp_var_373 = ((values[2] % 4));
        (_temp_var_373 == 0 ? indices.field_0 : (_temp_var_373 == 1 ? indices.field_1 : (_temp_var_373 == 2 ? indices.field_2 : (_temp_var_373 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_372 == 0 ? indices.field_0 : (_temp_var_372 == 1 ? indices.field_1 : (_temp_var_372 == 2 ? indices.field_2 : (_temp_var_372 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_371 == 0 ? indices.field_0 : (_temp_var_371 == 1 ? indices.field_1 : (_temp_var_371 == 2 ? indices.field_2 : (_temp_var_371 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_153(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_156)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_526;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_526 = _block_k_249_(_env_, _kernel_result_156[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_156[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_156[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_156[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_526 = 37;
    }
        
        _result_[_tid_] = temp_stencil_526;
    }
}



// TODO: There should be a better to check if _block_k_251_ is already defined
#ifndef _block_k_251__func
#define _block_k_251__func
__device__ int _block_k_251_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_374 = ((({ int _temp_var_375 = ((({ int _temp_var_376 = ((values[2] % 4));
        (_temp_var_376 == 0 ? indices.field_0 : (_temp_var_376 == 1 ? indices.field_1 : (_temp_var_376 == 2 ? indices.field_2 : (_temp_var_376 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_375 == 0 ? indices.field_0 : (_temp_var_375 == 1 ? indices.field_1 : (_temp_var_375 == 2 ? indices.field_2 : (_temp_var_375 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_374 == 0 ? indices.field_0 : (_temp_var_374 == 1 ? indices.field_1 : (_temp_var_374 == 2 ? indices.field_2 : (_temp_var_374 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_151(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_154)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_527;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_527 = _block_k_251_(_env_, _kernel_result_154[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_154[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_154[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_154[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_527 = 37;
    }
        
        _result_[_tid_] = temp_stencil_527;
    }
}



// TODO: There should be a better to check if _block_k_253_ is already defined
#ifndef _block_k_253__func
#define _block_k_253__func
__device__ int _block_k_253_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_377 = ((({ int _temp_var_378 = ((({ int _temp_var_379 = ((values[2] % 4));
        (_temp_var_379 == 0 ? indices.field_0 : (_temp_var_379 == 1 ? indices.field_1 : (_temp_var_379 == 2 ? indices.field_2 : (_temp_var_379 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_378 == 0 ? indices.field_0 : (_temp_var_378 == 1 ? indices.field_1 : (_temp_var_378 == 2 ? indices.field_2 : (_temp_var_378 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_377 == 0 ? indices.field_0 : (_temp_var_377 == 1 ? indices.field_1 : (_temp_var_377 == 2 ? indices.field_2 : (_temp_var_377 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_149(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_152)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_528;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_528 = _block_k_253_(_env_, _kernel_result_152[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_152[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_152[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_152[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_528 = 37;
    }
        
        _result_[_tid_] = temp_stencil_528;
    }
}



// TODO: There should be a better to check if _block_k_255_ is already defined
#ifndef _block_k_255__func
#define _block_k_255__func
__device__ int _block_k_255_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_380 = ((({ int _temp_var_381 = ((({ int _temp_var_382 = ((values[2] % 4));
        (_temp_var_382 == 0 ? indices.field_0 : (_temp_var_382 == 1 ? indices.field_1 : (_temp_var_382 == 2 ? indices.field_2 : (_temp_var_382 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_381 == 0 ? indices.field_0 : (_temp_var_381 == 1 ? indices.field_1 : (_temp_var_381 == 2 ? indices.field_2 : (_temp_var_381 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_380 == 0 ? indices.field_0 : (_temp_var_380 == 1 ? indices.field_1 : (_temp_var_380 == 2 ? indices.field_2 : (_temp_var_380 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_147(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_150)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_529;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_529 = _block_k_255_(_env_, _kernel_result_150[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_150[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_150[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_150[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_529 = 37;
    }
        
        _result_[_tid_] = temp_stencil_529;
    }
}



// TODO: There should be a better to check if _block_k_257_ is already defined
#ifndef _block_k_257__func
#define _block_k_257__func
__device__ int _block_k_257_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_383 = ((({ int _temp_var_384 = ((({ int _temp_var_385 = ((values[2] % 4));
        (_temp_var_385 == 0 ? indices.field_0 : (_temp_var_385 == 1 ? indices.field_1 : (_temp_var_385 == 2 ? indices.field_2 : (_temp_var_385 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_384 == 0 ? indices.field_0 : (_temp_var_384 == 1 ? indices.field_1 : (_temp_var_384 == 2 ? indices.field_2 : (_temp_var_384 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_383 == 0 ? indices.field_0 : (_temp_var_383 == 1 ? indices.field_1 : (_temp_var_383 == 2 ? indices.field_2 : (_temp_var_383 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_145(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_148)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_530;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_530 = _block_k_257_(_env_, _kernel_result_148[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_148[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_148[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_148[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_530 = 37;
    }
        
        _result_[_tid_] = temp_stencil_530;
    }
}



// TODO: There should be a better to check if _block_k_259_ is already defined
#ifndef _block_k_259__func
#define _block_k_259__func
__device__ int _block_k_259_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_386 = ((({ int _temp_var_387 = ((({ int _temp_var_388 = ((values[2] % 4));
        (_temp_var_388 == 0 ? indices.field_0 : (_temp_var_388 == 1 ? indices.field_1 : (_temp_var_388 == 2 ? indices.field_2 : (_temp_var_388 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_387 == 0 ? indices.field_0 : (_temp_var_387 == 1 ? indices.field_1 : (_temp_var_387 == 2 ? indices.field_2 : (_temp_var_387 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_386 == 0 ? indices.field_0 : (_temp_var_386 == 1 ? indices.field_1 : (_temp_var_386 == 2 ? indices.field_2 : (_temp_var_386 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_143(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_146)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_531;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_531 = _block_k_259_(_env_, _kernel_result_146[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_146[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_146[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_146[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_531 = 37;
    }
        
        _result_[_tid_] = temp_stencil_531;
    }
}



// TODO: There should be a better to check if _block_k_261_ is already defined
#ifndef _block_k_261__func
#define _block_k_261__func
__device__ int _block_k_261_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_389 = ((({ int _temp_var_390 = ((({ int _temp_var_391 = ((values[2] % 4));
        (_temp_var_391 == 0 ? indices.field_0 : (_temp_var_391 == 1 ? indices.field_1 : (_temp_var_391 == 2 ? indices.field_2 : (_temp_var_391 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_390 == 0 ? indices.field_0 : (_temp_var_390 == 1 ? indices.field_1 : (_temp_var_390 == 2 ? indices.field_2 : (_temp_var_390 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_389 == 0 ? indices.field_0 : (_temp_var_389 == 1 ? indices.field_1 : (_temp_var_389 == 2 ? indices.field_2 : (_temp_var_389 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_141(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_144)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_532;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_532 = _block_k_261_(_env_, _kernel_result_144[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_144[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_144[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_144[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_532 = 37;
    }
        
        _result_[_tid_] = temp_stencil_532;
    }
}



// TODO: There should be a better to check if _block_k_263_ is already defined
#ifndef _block_k_263__func
#define _block_k_263__func
__device__ int _block_k_263_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_392 = ((({ int _temp_var_393 = ((({ int _temp_var_394 = ((values[2] % 4));
        (_temp_var_394 == 0 ? indices.field_0 : (_temp_var_394 == 1 ? indices.field_1 : (_temp_var_394 == 2 ? indices.field_2 : (_temp_var_394 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_393 == 0 ? indices.field_0 : (_temp_var_393 == 1 ? indices.field_1 : (_temp_var_393 == 2 ? indices.field_2 : (_temp_var_393 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_392 == 0 ? indices.field_0 : (_temp_var_392 == 1 ? indices.field_1 : (_temp_var_392 == 2 ? indices.field_2 : (_temp_var_392 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_139(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_142)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_533;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_533 = _block_k_263_(_env_, _kernel_result_142[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_142[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_142[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_142[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_533 = 37;
    }
        
        _result_[_tid_] = temp_stencil_533;
    }
}



// TODO: There should be a better to check if _block_k_265_ is already defined
#ifndef _block_k_265__func
#define _block_k_265__func
__device__ int _block_k_265_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_395 = ((({ int _temp_var_396 = ((({ int _temp_var_397 = ((values[2] % 4));
        (_temp_var_397 == 0 ? indices.field_0 : (_temp_var_397 == 1 ? indices.field_1 : (_temp_var_397 == 2 ? indices.field_2 : (_temp_var_397 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_396 == 0 ? indices.field_0 : (_temp_var_396 == 1 ? indices.field_1 : (_temp_var_396 == 2 ? indices.field_2 : (_temp_var_396 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_395 == 0 ? indices.field_0 : (_temp_var_395 == 1 ? indices.field_1 : (_temp_var_395 == 2 ? indices.field_2 : (_temp_var_395 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_137(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_140)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_534;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_534 = _block_k_265_(_env_, _kernel_result_140[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_140[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_140[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_140[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_534 = 37;
    }
        
        _result_[_tid_] = temp_stencil_534;
    }
}



// TODO: There should be a better to check if _block_k_267_ is already defined
#ifndef _block_k_267__func
#define _block_k_267__func
__device__ int _block_k_267_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_398 = ((({ int _temp_var_399 = ((({ int _temp_var_400 = ((values[2] % 4));
        (_temp_var_400 == 0 ? indices.field_0 : (_temp_var_400 == 1 ? indices.field_1 : (_temp_var_400 == 2 ? indices.field_2 : (_temp_var_400 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_399 == 0 ? indices.field_0 : (_temp_var_399 == 1 ? indices.field_1 : (_temp_var_399 == 2 ? indices.field_2 : (_temp_var_399 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_398 == 0 ? indices.field_0 : (_temp_var_398 == 1 ? indices.field_1 : (_temp_var_398 == 2 ? indices.field_2 : (_temp_var_398 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_135(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_138)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_535;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_535 = _block_k_267_(_env_, _kernel_result_138[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_138[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_138[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_138[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_535 = 37;
    }
        
        _result_[_tid_] = temp_stencil_535;
    }
}



// TODO: There should be a better to check if _block_k_269_ is already defined
#ifndef _block_k_269__func
#define _block_k_269__func
__device__ int _block_k_269_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_401 = ((({ int _temp_var_402 = ((({ int _temp_var_403 = ((values[2] % 4));
        (_temp_var_403 == 0 ? indices.field_0 : (_temp_var_403 == 1 ? indices.field_1 : (_temp_var_403 == 2 ? indices.field_2 : (_temp_var_403 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_402 == 0 ? indices.field_0 : (_temp_var_402 == 1 ? indices.field_1 : (_temp_var_402 == 2 ? indices.field_2 : (_temp_var_402 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_401 == 0 ? indices.field_0 : (_temp_var_401 == 1 ? indices.field_1 : (_temp_var_401 == 2 ? indices.field_2 : (_temp_var_401 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_133(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_136)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_536;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_536 = _block_k_269_(_env_, _kernel_result_136[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_136[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_136[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_136[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_536 = 37;
    }
        
        _result_[_tid_] = temp_stencil_536;
    }
}



// TODO: There should be a better to check if _block_k_271_ is already defined
#ifndef _block_k_271__func
#define _block_k_271__func
__device__ int _block_k_271_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_404 = ((({ int _temp_var_405 = ((({ int _temp_var_406 = ((values[2] % 4));
        (_temp_var_406 == 0 ? indices.field_0 : (_temp_var_406 == 1 ? indices.field_1 : (_temp_var_406 == 2 ? indices.field_2 : (_temp_var_406 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_405 == 0 ? indices.field_0 : (_temp_var_405 == 1 ? indices.field_1 : (_temp_var_405 == 2 ? indices.field_2 : (_temp_var_405 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_404 == 0 ? indices.field_0 : (_temp_var_404 == 1 ? indices.field_1 : (_temp_var_404 == 2 ? indices.field_2 : (_temp_var_404 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_131(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_134)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_537;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_537 = _block_k_271_(_env_, _kernel_result_134[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_134[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_134[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_134[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_537 = 37;
    }
        
        _result_[_tid_] = temp_stencil_537;
    }
}



// TODO: There should be a better to check if _block_k_273_ is already defined
#ifndef _block_k_273__func
#define _block_k_273__func
__device__ int _block_k_273_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_407 = ((({ int _temp_var_408 = ((({ int _temp_var_409 = ((values[2] % 4));
        (_temp_var_409 == 0 ? indices.field_0 : (_temp_var_409 == 1 ? indices.field_1 : (_temp_var_409 == 2 ? indices.field_2 : (_temp_var_409 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_408 == 0 ? indices.field_0 : (_temp_var_408 == 1 ? indices.field_1 : (_temp_var_408 == 2 ? indices.field_2 : (_temp_var_408 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_407 == 0 ? indices.field_0 : (_temp_var_407 == 1 ? indices.field_1 : (_temp_var_407 == 2 ? indices.field_2 : (_temp_var_407 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_129(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_132)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_538;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_538 = _block_k_273_(_env_, _kernel_result_132[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_132[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_132[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_132[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_538 = 37;
    }
        
        _result_[_tid_] = temp_stencil_538;
    }
}



// TODO: There should be a better to check if _block_k_275_ is already defined
#ifndef _block_k_275__func
#define _block_k_275__func
__device__ int _block_k_275_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_410 = ((({ int _temp_var_411 = ((({ int _temp_var_412 = ((values[2] % 4));
        (_temp_var_412 == 0 ? indices.field_0 : (_temp_var_412 == 1 ? indices.field_1 : (_temp_var_412 == 2 ? indices.field_2 : (_temp_var_412 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_411 == 0 ? indices.field_0 : (_temp_var_411 == 1 ? indices.field_1 : (_temp_var_411 == 2 ? indices.field_2 : (_temp_var_411 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_410 == 0 ? indices.field_0 : (_temp_var_410 == 1 ? indices.field_1 : (_temp_var_410 == 2 ? indices.field_2 : (_temp_var_410 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_127(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_130)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_539;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_539 = _block_k_275_(_env_, _kernel_result_130[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_130[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_130[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_130[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_539 = 37;
    }
        
        _result_[_tid_] = temp_stencil_539;
    }
}



// TODO: There should be a better to check if _block_k_277_ is already defined
#ifndef _block_k_277__func
#define _block_k_277__func
__device__ int _block_k_277_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_413 = ((({ int _temp_var_414 = ((({ int _temp_var_415 = ((values[2] % 4));
        (_temp_var_415 == 0 ? indices.field_0 : (_temp_var_415 == 1 ? indices.field_1 : (_temp_var_415 == 2 ? indices.field_2 : (_temp_var_415 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_414 == 0 ? indices.field_0 : (_temp_var_414 == 1 ? indices.field_1 : (_temp_var_414 == 2 ? indices.field_2 : (_temp_var_414 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_413 == 0 ? indices.field_0 : (_temp_var_413 == 1 ? indices.field_1 : (_temp_var_413 == 2 ? indices.field_2 : (_temp_var_413 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_125(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_128)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_540;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_540 = _block_k_277_(_env_, _kernel_result_128[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_128[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_128[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_128[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_540 = 37;
    }
        
        _result_[_tid_] = temp_stencil_540;
    }
}



// TODO: There should be a better to check if _block_k_279_ is already defined
#ifndef _block_k_279__func
#define _block_k_279__func
__device__ int _block_k_279_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_416 = ((({ int _temp_var_417 = ((({ int _temp_var_418 = ((values[2] % 4));
        (_temp_var_418 == 0 ? indices.field_0 : (_temp_var_418 == 1 ? indices.field_1 : (_temp_var_418 == 2 ? indices.field_2 : (_temp_var_418 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_417 == 0 ? indices.field_0 : (_temp_var_417 == 1 ? indices.field_1 : (_temp_var_417 == 2 ? indices.field_2 : (_temp_var_417 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_416 == 0 ? indices.field_0 : (_temp_var_416 == 1 ? indices.field_1 : (_temp_var_416 == 2 ? indices.field_2 : (_temp_var_416 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_123(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_126)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_541;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_541 = _block_k_279_(_env_, _kernel_result_126[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_126[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_126[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_126[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_541 = 37;
    }
        
        _result_[_tid_] = temp_stencil_541;
    }
}



// TODO: There should be a better to check if _block_k_281_ is already defined
#ifndef _block_k_281__func
#define _block_k_281__func
__device__ int _block_k_281_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_419 = ((({ int _temp_var_420 = ((({ int _temp_var_421 = ((values[2] % 4));
        (_temp_var_421 == 0 ? indices.field_0 : (_temp_var_421 == 1 ? indices.field_1 : (_temp_var_421 == 2 ? indices.field_2 : (_temp_var_421 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_420 == 0 ? indices.field_0 : (_temp_var_420 == 1 ? indices.field_1 : (_temp_var_420 == 2 ? indices.field_2 : (_temp_var_420 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_419 == 0 ? indices.field_0 : (_temp_var_419 == 1 ? indices.field_1 : (_temp_var_419 == 2 ? indices.field_2 : (_temp_var_419 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_121(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_124)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_542;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_542 = _block_k_281_(_env_, _kernel_result_124[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_124[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_124[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_124[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_542 = 37;
    }
        
        _result_[_tid_] = temp_stencil_542;
    }
}



// TODO: There should be a better to check if _block_k_283_ is already defined
#ifndef _block_k_283__func
#define _block_k_283__func
__device__ int _block_k_283_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_422 = ((({ int _temp_var_423 = ((({ int _temp_var_424 = ((values[2] % 4));
        (_temp_var_424 == 0 ? indices.field_0 : (_temp_var_424 == 1 ? indices.field_1 : (_temp_var_424 == 2 ? indices.field_2 : (_temp_var_424 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_423 == 0 ? indices.field_0 : (_temp_var_423 == 1 ? indices.field_1 : (_temp_var_423 == 2 ? indices.field_2 : (_temp_var_423 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_422 == 0 ? indices.field_0 : (_temp_var_422 == 1 ? indices.field_1 : (_temp_var_422 == 2 ? indices.field_2 : (_temp_var_422 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_119(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_122)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_543;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_543 = _block_k_283_(_env_, _kernel_result_122[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_122[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_122[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_122[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_543 = 37;
    }
        
        _result_[_tid_] = temp_stencil_543;
    }
}



// TODO: There should be a better to check if _block_k_285_ is already defined
#ifndef _block_k_285__func
#define _block_k_285__func
__device__ int _block_k_285_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_425 = ((({ int _temp_var_426 = ((({ int _temp_var_427 = ((values[2] % 4));
        (_temp_var_427 == 0 ? indices.field_0 : (_temp_var_427 == 1 ? indices.field_1 : (_temp_var_427 == 2 ? indices.field_2 : (_temp_var_427 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_426 == 0 ? indices.field_0 : (_temp_var_426 == 1 ? indices.field_1 : (_temp_var_426 == 2 ? indices.field_2 : (_temp_var_426 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_425 == 0 ? indices.field_0 : (_temp_var_425 == 1 ? indices.field_1 : (_temp_var_425 == 2 ? indices.field_2 : (_temp_var_425 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_117(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_120)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_544;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_544 = _block_k_285_(_env_, _kernel_result_120[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_120[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_120[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_120[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_544 = 37;
    }
        
        _result_[_tid_] = temp_stencil_544;
    }
}



// TODO: There should be a better to check if _block_k_287_ is already defined
#ifndef _block_k_287__func
#define _block_k_287__func
__device__ int _block_k_287_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_428 = ((({ int _temp_var_429 = ((({ int _temp_var_430 = ((values[2] % 4));
        (_temp_var_430 == 0 ? indices.field_0 : (_temp_var_430 == 1 ? indices.field_1 : (_temp_var_430 == 2 ? indices.field_2 : (_temp_var_430 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_429 == 0 ? indices.field_0 : (_temp_var_429 == 1 ? indices.field_1 : (_temp_var_429 == 2 ? indices.field_2 : (_temp_var_429 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_428 == 0 ? indices.field_0 : (_temp_var_428 == 1 ? indices.field_1 : (_temp_var_428 == 2 ? indices.field_2 : (_temp_var_428 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_115(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_118)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_545;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_545 = _block_k_287_(_env_, _kernel_result_118[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_118[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_118[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_118[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_545 = 37;
    }
        
        _result_[_tid_] = temp_stencil_545;
    }
}



// TODO: There should be a better to check if _block_k_289_ is already defined
#ifndef _block_k_289__func
#define _block_k_289__func
__device__ int _block_k_289_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_431 = ((({ int _temp_var_432 = ((({ int _temp_var_433 = ((values[2] % 4));
        (_temp_var_433 == 0 ? indices.field_0 : (_temp_var_433 == 1 ? indices.field_1 : (_temp_var_433 == 2 ? indices.field_2 : (_temp_var_433 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_432 == 0 ? indices.field_0 : (_temp_var_432 == 1 ? indices.field_1 : (_temp_var_432 == 2 ? indices.field_2 : (_temp_var_432 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_431 == 0 ? indices.field_0 : (_temp_var_431 == 1 ? indices.field_1 : (_temp_var_431 == 2 ? indices.field_2 : (_temp_var_431 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_113(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_116)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_546;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_546 = _block_k_289_(_env_, _kernel_result_116[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_116[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_116[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_116[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_546 = 37;
    }
        
        _result_[_tid_] = temp_stencil_546;
    }
}



// TODO: There should be a better to check if _block_k_291_ is already defined
#ifndef _block_k_291__func
#define _block_k_291__func
__device__ int _block_k_291_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_434 = ((({ int _temp_var_435 = ((({ int _temp_var_436 = ((values[2] % 4));
        (_temp_var_436 == 0 ? indices.field_0 : (_temp_var_436 == 1 ? indices.field_1 : (_temp_var_436 == 2 ? indices.field_2 : (_temp_var_436 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_435 == 0 ? indices.field_0 : (_temp_var_435 == 1 ? indices.field_1 : (_temp_var_435 == 2 ? indices.field_2 : (_temp_var_435 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_434 == 0 ? indices.field_0 : (_temp_var_434 == 1 ? indices.field_1 : (_temp_var_434 == 2 ? indices.field_2 : (_temp_var_434 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_111(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_114)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_547;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_547 = _block_k_291_(_env_, _kernel_result_114[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_114[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_114[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_114[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_547 = 37;
    }
        
        _result_[_tid_] = temp_stencil_547;
    }
}



// TODO: There should be a better to check if _block_k_293_ is already defined
#ifndef _block_k_293__func
#define _block_k_293__func
__device__ int _block_k_293_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_437 = ((({ int _temp_var_438 = ((({ int _temp_var_439 = ((values[2] % 4));
        (_temp_var_439 == 0 ? indices.field_0 : (_temp_var_439 == 1 ? indices.field_1 : (_temp_var_439 == 2 ? indices.field_2 : (_temp_var_439 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_438 == 0 ? indices.field_0 : (_temp_var_438 == 1 ? indices.field_1 : (_temp_var_438 == 2 ? indices.field_2 : (_temp_var_438 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_437 == 0 ? indices.field_0 : (_temp_var_437 == 1 ? indices.field_1 : (_temp_var_437 == 2 ? indices.field_2 : (_temp_var_437 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_109(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_112)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_548;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_548 = _block_k_293_(_env_, _kernel_result_112[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_112[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_112[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_112[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_548 = 37;
    }
        
        _result_[_tid_] = temp_stencil_548;
    }
}



// TODO: There should be a better to check if _block_k_295_ is already defined
#ifndef _block_k_295__func
#define _block_k_295__func
__device__ int _block_k_295_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_440 = ((({ int _temp_var_441 = ((({ int _temp_var_442 = ((values[2] % 4));
        (_temp_var_442 == 0 ? indices.field_0 : (_temp_var_442 == 1 ? indices.field_1 : (_temp_var_442 == 2 ? indices.field_2 : (_temp_var_442 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_441 == 0 ? indices.field_0 : (_temp_var_441 == 1 ? indices.field_1 : (_temp_var_441 == 2 ? indices.field_2 : (_temp_var_441 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_440 == 0 ? indices.field_0 : (_temp_var_440 == 1 ? indices.field_1 : (_temp_var_440 == 2 ? indices.field_2 : (_temp_var_440 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_107(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_110)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_549;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_549 = _block_k_295_(_env_, _kernel_result_110[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_110[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_110[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_110[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_549 = 37;
    }
        
        _result_[_tid_] = temp_stencil_549;
    }
}



// TODO: There should be a better to check if _block_k_297_ is already defined
#ifndef _block_k_297__func
#define _block_k_297__func
__device__ int _block_k_297_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_443 = ((({ int _temp_var_444 = ((({ int _temp_var_445 = ((values[2] % 4));
        (_temp_var_445 == 0 ? indices.field_0 : (_temp_var_445 == 1 ? indices.field_1 : (_temp_var_445 == 2 ? indices.field_2 : (_temp_var_445 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_444 == 0 ? indices.field_0 : (_temp_var_444 == 1 ? indices.field_1 : (_temp_var_444 == 2 ? indices.field_2 : (_temp_var_444 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_443 == 0 ? indices.field_0 : (_temp_var_443 == 1 ? indices.field_1 : (_temp_var_443 == 2 ? indices.field_2 : (_temp_var_443 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_105(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_108)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_550;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_550 = _block_k_297_(_env_, _kernel_result_108[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_108[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_108[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_108[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_550 = 37;
    }
        
        _result_[_tid_] = temp_stencil_550;
    }
}



// TODO: There should be a better to check if _block_k_299_ is already defined
#ifndef _block_k_299__func
#define _block_k_299__func
__device__ int _block_k_299_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_446 = ((({ int _temp_var_447 = ((({ int _temp_var_448 = ((values[2] % 4));
        (_temp_var_448 == 0 ? indices.field_0 : (_temp_var_448 == 1 ? indices.field_1 : (_temp_var_448 == 2 ? indices.field_2 : (_temp_var_448 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_447 == 0 ? indices.field_0 : (_temp_var_447 == 1 ? indices.field_1 : (_temp_var_447 == 2 ? indices.field_2 : (_temp_var_447 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_446 == 0 ? indices.field_0 : (_temp_var_446 == 1 ? indices.field_1 : (_temp_var_446 == 2 ? indices.field_2 : (_temp_var_446 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_103(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_106)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_551;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_551 = _block_k_299_(_env_, _kernel_result_106[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_106[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_106[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_106[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_551 = 37;
    }
        
        _result_[_tid_] = temp_stencil_551;
    }
}



// TODO: There should be a better to check if _block_k_301_ is already defined
#ifndef _block_k_301__func
#define _block_k_301__func
__device__ int _block_k_301_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_449 = ((({ int _temp_var_450 = ((({ int _temp_var_451 = ((values[2] % 4));
        (_temp_var_451 == 0 ? indices.field_0 : (_temp_var_451 == 1 ? indices.field_1 : (_temp_var_451 == 2 ? indices.field_2 : (_temp_var_451 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_450 == 0 ? indices.field_0 : (_temp_var_450 == 1 ? indices.field_1 : (_temp_var_450 == 2 ? indices.field_2 : (_temp_var_450 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_449 == 0 ? indices.field_0 : (_temp_var_449 == 1 ? indices.field_1 : (_temp_var_449 == 2 ? indices.field_2 : (_temp_var_449 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_101(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_104)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_552;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_552 = _block_k_301_(_env_, _kernel_result_104[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_104[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_104[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_104[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_552 = 37;
    }
        
        _result_[_tid_] = temp_stencil_552;
    }
}



// TODO: There should be a better to check if _block_k_303_ is already defined
#ifndef _block_k_303__func
#define _block_k_303__func
__device__ int _block_k_303_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_452 = ((({ int _temp_var_453 = ((({ int _temp_var_454 = ((values[2] % 4));
        (_temp_var_454 == 0 ? indices.field_0 : (_temp_var_454 == 1 ? indices.field_1 : (_temp_var_454 == 2 ? indices.field_2 : (_temp_var_454 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_453 == 0 ? indices.field_0 : (_temp_var_453 == 1 ? indices.field_1 : (_temp_var_453 == 2 ? indices.field_2 : (_temp_var_453 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_452 == 0 ? indices.field_0 : (_temp_var_452 == 1 ? indices.field_1 : (_temp_var_452 == 2 ? indices.field_2 : (_temp_var_452 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_99(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_102)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_553;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_553 = _block_k_303_(_env_, _kernel_result_102[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_102[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_102[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_102[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_553 = 37;
    }
        
        _result_[_tid_] = temp_stencil_553;
    }
}



// TODO: There should be a better to check if _block_k_305_ is already defined
#ifndef _block_k_305__func
#define _block_k_305__func
__device__ int _block_k_305_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_455 = ((({ int _temp_var_456 = ((({ int _temp_var_457 = ((values[2] % 4));
        (_temp_var_457 == 0 ? indices.field_0 : (_temp_var_457 == 1 ? indices.field_1 : (_temp_var_457 == 2 ? indices.field_2 : (_temp_var_457 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_456 == 0 ? indices.field_0 : (_temp_var_456 == 1 ? indices.field_1 : (_temp_var_456 == 2 ? indices.field_2 : (_temp_var_456 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_455 == 0 ? indices.field_0 : (_temp_var_455 == 1 ? indices.field_1 : (_temp_var_455 == 2 ? indices.field_2 : (_temp_var_455 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_97(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_100)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_554;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_554 = _block_k_305_(_env_, _kernel_result_100[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_100[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_100[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_100[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_554 = 37;
    }
        
        _result_[_tid_] = temp_stencil_554;
    }
}



// TODO: There should be a better to check if _block_k_307_ is already defined
#ifndef _block_k_307__func
#define _block_k_307__func
__device__ int _block_k_307_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_458 = ((({ int _temp_var_459 = ((({ int _temp_var_460 = ((values[2] % 4));
        (_temp_var_460 == 0 ? indices.field_0 : (_temp_var_460 == 1 ? indices.field_1 : (_temp_var_460 == 2 ? indices.field_2 : (_temp_var_460 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_459 == 0 ? indices.field_0 : (_temp_var_459 == 1 ? indices.field_1 : (_temp_var_459 == 2 ? indices.field_2 : (_temp_var_459 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_458 == 0 ? indices.field_0 : (_temp_var_458 == 1 ? indices.field_1 : (_temp_var_458 == 2 ? indices.field_2 : (_temp_var_458 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_95(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_98)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_555;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_555 = _block_k_307_(_env_, _kernel_result_98[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_98[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_98[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_98[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_555 = 37;
    }
        
        _result_[_tid_] = temp_stencil_555;
    }
}



// TODO: There should be a better to check if _block_k_309_ is already defined
#ifndef _block_k_309__func
#define _block_k_309__func
__device__ int _block_k_309_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_461 = ((({ int _temp_var_462 = ((({ int _temp_var_463 = ((values[2] % 4));
        (_temp_var_463 == 0 ? indices.field_0 : (_temp_var_463 == 1 ? indices.field_1 : (_temp_var_463 == 2 ? indices.field_2 : (_temp_var_463 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_462 == 0 ? indices.field_0 : (_temp_var_462 == 1 ? indices.field_1 : (_temp_var_462 == 2 ? indices.field_2 : (_temp_var_462 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_461 == 0 ? indices.field_0 : (_temp_var_461 == 1 ? indices.field_1 : (_temp_var_461 == 2 ? indices.field_2 : (_temp_var_461 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_93(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_96)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_556;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_556 = _block_k_309_(_env_, _kernel_result_96[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_96[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_96[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_96[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_556 = 37;
    }
        
        _result_[_tid_] = temp_stencil_556;
    }
}



// TODO: There should be a better to check if _block_k_311_ is already defined
#ifndef _block_k_311__func
#define _block_k_311__func
__device__ int _block_k_311_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_464 = ((({ int _temp_var_465 = ((({ int _temp_var_466 = ((values[2] % 4));
        (_temp_var_466 == 0 ? indices.field_0 : (_temp_var_466 == 1 ? indices.field_1 : (_temp_var_466 == 2 ? indices.field_2 : (_temp_var_466 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_465 == 0 ? indices.field_0 : (_temp_var_465 == 1 ? indices.field_1 : (_temp_var_465 == 2 ? indices.field_2 : (_temp_var_465 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_464 == 0 ? indices.field_0 : (_temp_var_464 == 1 ? indices.field_1 : (_temp_var_464 == 2 ? indices.field_2 : (_temp_var_464 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_91(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_94)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_557;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_557 = _block_k_311_(_env_, _kernel_result_94[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_94[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_94[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_94[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_557 = 37;
    }
        
        _result_[_tid_] = temp_stencil_557;
    }
}



// TODO: There should be a better to check if _block_k_313_ is already defined
#ifndef _block_k_313__func
#define _block_k_313__func
__device__ int _block_k_313_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_467 = ((({ int _temp_var_468 = ((({ int _temp_var_469 = ((values[2] % 4));
        (_temp_var_469 == 0 ? indices.field_0 : (_temp_var_469 == 1 ? indices.field_1 : (_temp_var_469 == 2 ? indices.field_2 : (_temp_var_469 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_468 == 0 ? indices.field_0 : (_temp_var_468 == 1 ? indices.field_1 : (_temp_var_468 == 2 ? indices.field_2 : (_temp_var_468 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_467 == 0 ? indices.field_0 : (_temp_var_467 == 1 ? indices.field_1 : (_temp_var_467 == 2 ? indices.field_2 : (_temp_var_467 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_89(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_92)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_558;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_558 = _block_k_313_(_env_, _kernel_result_92[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_92[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_92[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_92[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_558 = 37;
    }
        
        _result_[_tid_] = temp_stencil_558;
    }
}



// TODO: There should be a better to check if _block_k_315_ is already defined
#ifndef _block_k_315__func
#define _block_k_315__func
__device__ int _block_k_315_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_470 = ((({ int _temp_var_471 = ((({ int _temp_var_472 = ((values[2] % 4));
        (_temp_var_472 == 0 ? indices.field_0 : (_temp_var_472 == 1 ? indices.field_1 : (_temp_var_472 == 2 ? indices.field_2 : (_temp_var_472 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_471 == 0 ? indices.field_0 : (_temp_var_471 == 1 ? indices.field_1 : (_temp_var_471 == 2 ? indices.field_2 : (_temp_var_471 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_470 == 0 ? indices.field_0 : (_temp_var_470 == 1 ? indices.field_1 : (_temp_var_470 == 2 ? indices.field_2 : (_temp_var_470 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_87(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_90)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_559;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_559 = _block_k_315_(_env_, _kernel_result_90[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_90[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_90[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_90[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_559 = 37;
    }
        
        _result_[_tid_] = temp_stencil_559;
    }
}



// TODO: There should be a better to check if _block_k_317_ is already defined
#ifndef _block_k_317__func
#define _block_k_317__func
__device__ int _block_k_317_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_473 = ((({ int _temp_var_474 = ((({ int _temp_var_475 = ((values[2] % 4));
        (_temp_var_475 == 0 ? indices.field_0 : (_temp_var_475 == 1 ? indices.field_1 : (_temp_var_475 == 2 ? indices.field_2 : (_temp_var_475 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_474 == 0 ? indices.field_0 : (_temp_var_474 == 1 ? indices.field_1 : (_temp_var_474 == 2 ? indices.field_2 : (_temp_var_474 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_473 == 0 ? indices.field_0 : (_temp_var_473 == 1 ? indices.field_1 : (_temp_var_473 == 2 ? indices.field_2 : (_temp_var_473 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_85(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_88)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_560;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_560 = _block_k_317_(_env_, _kernel_result_88[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_88[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_88[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_88[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_560 = 37;
    }
        
        _result_[_tid_] = temp_stencil_560;
    }
}



// TODO: There should be a better to check if _block_k_319_ is already defined
#ifndef _block_k_319__func
#define _block_k_319__func
__device__ int _block_k_319_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_476 = ((({ int _temp_var_477 = ((({ int _temp_var_478 = ((values[2] % 4));
        (_temp_var_478 == 0 ? indices.field_0 : (_temp_var_478 == 1 ? indices.field_1 : (_temp_var_478 == 2 ? indices.field_2 : (_temp_var_478 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_477 == 0 ? indices.field_0 : (_temp_var_477 == 1 ? indices.field_1 : (_temp_var_477 == 2 ? indices.field_2 : (_temp_var_477 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_476 == 0 ? indices.field_0 : (_temp_var_476 == 1 ? indices.field_1 : (_temp_var_476 == 2 ? indices.field_2 : (_temp_var_476 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_83(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_86)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_561;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_561 = _block_k_319_(_env_, _kernel_result_86[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_86[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_86[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_86[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_561 = 37;
    }
        
        _result_[_tid_] = temp_stencil_561;
    }
}



// TODO: There should be a better to check if _block_k_321_ is already defined
#ifndef _block_k_321__func
#define _block_k_321__func
__device__ int _block_k_321_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_479 = ((({ int _temp_var_480 = ((({ int _temp_var_481 = ((values[2] % 4));
        (_temp_var_481 == 0 ? indices.field_0 : (_temp_var_481 == 1 ? indices.field_1 : (_temp_var_481 == 2 ? indices.field_2 : (_temp_var_481 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_480 == 0 ? indices.field_0 : (_temp_var_480 == 1 ? indices.field_1 : (_temp_var_480 == 2 ? indices.field_2 : (_temp_var_480 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_479 == 0 ? indices.field_0 : (_temp_var_479 == 1 ? indices.field_1 : (_temp_var_479 == 2 ? indices.field_2 : (_temp_var_479 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_81(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_84)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_562;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_562 = _block_k_321_(_env_, _kernel_result_84[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_84[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_84[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_84[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_562 = 37;
    }
        
        _result_[_tid_] = temp_stencil_562;
    }
}



// TODO: There should be a better to check if _block_k_323_ is already defined
#ifndef _block_k_323__func
#define _block_k_323__func
__device__ int _block_k_323_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_482 = ((({ int _temp_var_483 = ((({ int _temp_var_484 = ((values[2] % 4));
        (_temp_var_484 == 0 ? indices.field_0 : (_temp_var_484 == 1 ? indices.field_1 : (_temp_var_484 == 2 ? indices.field_2 : (_temp_var_484 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_483 == 0 ? indices.field_0 : (_temp_var_483 == 1 ? indices.field_1 : (_temp_var_483 == 2 ? indices.field_2 : (_temp_var_483 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_482 == 0 ? indices.field_0 : (_temp_var_482 == 1 ? indices.field_1 : (_temp_var_482 == 2 ? indices.field_2 : (_temp_var_482 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_79(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_82)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_563;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_563 = _block_k_323_(_env_, _kernel_result_82[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_82[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_82[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_82[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_563 = 37;
    }
        
        _result_[_tid_] = temp_stencil_563;
    }
}



// TODO: There should be a better to check if _block_k_325_ is already defined
#ifndef _block_k_325__func
#define _block_k_325__func
__device__ int _block_k_325_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_485 = ((({ int _temp_var_486 = ((({ int _temp_var_487 = ((values[2] % 4));
        (_temp_var_487 == 0 ? indices.field_0 : (_temp_var_487 == 1 ? indices.field_1 : (_temp_var_487 == 2 ? indices.field_2 : (_temp_var_487 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_486 == 0 ? indices.field_0 : (_temp_var_486 == 1 ? indices.field_1 : (_temp_var_486 == 2 ? indices.field_2 : (_temp_var_486 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_485 == 0 ? indices.field_0 : (_temp_var_485 == 1 ? indices.field_1 : (_temp_var_485 == 2 ? indices.field_2 : (_temp_var_485 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_77(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_80)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_564;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_564 = _block_k_325_(_env_, _kernel_result_80[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_80[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_80[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_80[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_564 = 37;
    }
        
        _result_[_tid_] = temp_stencil_564;
    }
}



// TODO: There should be a better to check if _block_k_327_ is already defined
#ifndef _block_k_327__func
#define _block_k_327__func
__device__ int _block_k_327_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_488 = ((({ int _temp_var_489 = ((({ int _temp_var_490 = ((values[2] % 4));
        (_temp_var_490 == 0 ? indices.field_0 : (_temp_var_490 == 1 ? indices.field_1 : (_temp_var_490 == 2 ? indices.field_2 : (_temp_var_490 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_489 == 0 ? indices.field_0 : (_temp_var_489 == 1 ? indices.field_1 : (_temp_var_489 == 2 ? indices.field_2 : (_temp_var_489 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_488 == 0 ? indices.field_0 : (_temp_var_488 == 1 ? indices.field_1 : (_temp_var_488 == 2 ? indices.field_2 : (_temp_var_488 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_75(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_78)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_565;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_565 = _block_k_327_(_env_, _kernel_result_78[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_78[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_78[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_78[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_565 = 37;
    }
        
        _result_[_tid_] = temp_stencil_565;
    }
}



// TODO: There should be a better to check if _block_k_329_ is already defined
#ifndef _block_k_329__func
#define _block_k_329__func
__device__ int _block_k_329_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_491 = ((({ int _temp_var_492 = ((({ int _temp_var_493 = ((values[2] % 4));
        (_temp_var_493 == 0 ? indices.field_0 : (_temp_var_493 == 1 ? indices.field_1 : (_temp_var_493 == 2 ? indices.field_2 : (_temp_var_493 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_492 == 0 ? indices.field_0 : (_temp_var_492 == 1 ? indices.field_1 : (_temp_var_492 == 2 ? indices.field_2 : (_temp_var_492 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_491 == 0 ? indices.field_0 : (_temp_var_491 == 1 ? indices.field_1 : (_temp_var_491 == 2 ? indices.field_2 : (_temp_var_491 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_73(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_76)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_566;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_566 = _block_k_329_(_env_, _kernel_result_76[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_76[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_76[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_76[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_566 = 37;
    }
        
        _result_[_tid_] = temp_stencil_566;
    }
}



// TODO: There should be a better to check if _block_k_331_ is already defined
#ifndef _block_k_331__func
#define _block_k_331__func
__device__ int _block_k_331_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_494 = ((({ int _temp_var_495 = ((({ int _temp_var_496 = ((values[2] % 4));
        (_temp_var_496 == 0 ? indices.field_0 : (_temp_var_496 == 1 ? indices.field_1 : (_temp_var_496 == 2 ? indices.field_2 : (_temp_var_496 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_495 == 0 ? indices.field_0 : (_temp_var_495 == 1 ? indices.field_1 : (_temp_var_495 == 2 ? indices.field_2 : (_temp_var_495 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_494 == 0 ? indices.field_0 : (_temp_var_494 == 1 ? indices.field_1 : (_temp_var_494 == 2 ? indices.field_2 : (_temp_var_494 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_71(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_74)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_567;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_567 = _block_k_331_(_env_, _kernel_result_74[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_74[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_74[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_74[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_567 = 37;
    }
        
        _result_[_tid_] = temp_stencil_567;
    }
}



// TODO: There should be a better to check if _block_k_333_ is already defined
#ifndef _block_k_333__func
#define _block_k_333__func
__device__ int _block_k_333_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_497 = ((({ int _temp_var_498 = ((({ int _temp_var_499 = ((values[2] % 4));
        (_temp_var_499 == 0 ? indices.field_0 : (_temp_var_499 == 1 ? indices.field_1 : (_temp_var_499 == 2 ? indices.field_2 : (_temp_var_499 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_498 == 0 ? indices.field_0 : (_temp_var_498 == 1 ? indices.field_1 : (_temp_var_498 == 2 ? indices.field_2 : (_temp_var_498 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_497 == 0 ? indices.field_0 : (_temp_var_497 == 1 ? indices.field_1 : (_temp_var_497 == 2 ? indices.field_2 : (_temp_var_497 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_69(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_72)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_568;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_568 = _block_k_333_(_env_, _kernel_result_72[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_72[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_72[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_72[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_568 = 37;
    }
        
        _result_[_tid_] = temp_stencil_568;
    }
}



// TODO: There should be a better to check if _block_k_335_ is already defined
#ifndef _block_k_335__func
#define _block_k_335__func
__device__ int _block_k_335_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_500 = ((({ int _temp_var_501 = ((({ int _temp_var_502 = ((values[2] % 4));
        (_temp_var_502 == 0 ? indices.field_0 : (_temp_var_502 == 1 ? indices.field_1 : (_temp_var_502 == 2 ? indices.field_2 : (_temp_var_502 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_501 == 0 ? indices.field_0 : (_temp_var_501 == 1 ? indices.field_1 : (_temp_var_501 == 2 ? indices.field_2 : (_temp_var_501 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_500 == 0 ? indices.field_0 : (_temp_var_500 == 1 ? indices.field_1 : (_temp_var_500 == 2 ? indices.field_2 : (_temp_var_500 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_67(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_70)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_569;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_569 = _block_k_335_(_env_, _kernel_result_70[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_70[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_70[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_70[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_569 = 37;
    }
        
        _result_[_tid_] = temp_stencil_569;
    }
}



// TODO: There should be a better to check if _block_k_337_ is already defined
#ifndef _block_k_337__func
#define _block_k_337__func
__device__ int _block_k_337_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_503 = ((({ int _temp_var_504 = ((({ int _temp_var_505 = ((values[2] % 4));
        (_temp_var_505 == 0 ? indices.field_0 : (_temp_var_505 == 1 ? indices.field_1 : (_temp_var_505 == 2 ? indices.field_2 : (_temp_var_505 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_504 == 0 ? indices.field_0 : (_temp_var_504 == 1 ? indices.field_1 : (_temp_var_504 == 2 ? indices.field_2 : (_temp_var_504 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_503 == 0 ? indices.field_0 : (_temp_var_503 == 1 ? indices.field_1 : (_temp_var_503 == 2 ? indices.field_2 : (_temp_var_503 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_65(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_68)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_570;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_570 = _block_k_337_(_env_, _kernel_result_68[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_68[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_68[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_68[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_570 = 37;
    }
        
        _result_[_tid_] = temp_stencil_570;
    }
}



// TODO: There should be a better to check if _block_k_339_ is already defined
#ifndef _block_k_339__func
#define _block_k_339__func
__device__ int _block_k_339_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_506 = ((({ int _temp_var_507 = ((({ int _temp_var_508 = ((values[2] % 4));
        (_temp_var_508 == 0 ? indices.field_0 : (_temp_var_508 == 1 ? indices.field_1 : (_temp_var_508 == 2 ? indices.field_2 : (_temp_var_508 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_507 == 0 ? indices.field_0 : (_temp_var_507 == 1 ? indices.field_1 : (_temp_var_507 == 2 ? indices.field_2 : (_temp_var_507 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_506 == 0 ? indices.field_0 : (_temp_var_506 == 1 ? indices.field_1 : (_temp_var_506 == 2 ? indices.field_2 : (_temp_var_506 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_63(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_66)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_571;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_571 = _block_k_339_(_env_, _kernel_result_66[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_66[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_66[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_66[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_571 = 37;
    }
        
        _result_[_tid_] = temp_stencil_571;
    }
}



// TODO: There should be a better to check if _block_k_341_ is already defined
#ifndef _block_k_341__func
#define _block_k_341__func
__device__ int _block_k_341_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_509 = ((({ int _temp_var_510 = ((({ int _temp_var_511 = ((values[2] % 4));
        (_temp_var_511 == 0 ? indices.field_0 : (_temp_var_511 == 1 ? indices.field_1 : (_temp_var_511 == 2 ? indices.field_2 : (_temp_var_511 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_510 == 0 ? indices.field_0 : (_temp_var_510 == 1 ? indices.field_1 : (_temp_var_510 == 2 ? indices.field_2 : (_temp_var_510 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_509 == 0 ? indices.field_0 : (_temp_var_509 == 1 ? indices.field_1 : (_temp_var_509 == 2 ? indices.field_2 : (_temp_var_509 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_61(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_64)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_572;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_572 = _block_k_341_(_env_, _kernel_result_64[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_64[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_64[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_64[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_572 = 37;
    }
        
        _result_[_tid_] = temp_stencil_572;
    }
}



// TODO: There should be a better to check if _block_k_343_ is already defined
#ifndef _block_k_343__func
#define _block_k_343__func
__device__ int _block_k_343_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_512 = ((({ int _temp_var_513 = ((({ int _temp_var_514 = ((values[2] % 4));
        (_temp_var_514 == 0 ? indices.field_0 : (_temp_var_514 == 1 ? indices.field_1 : (_temp_var_514 == 2 ? indices.field_2 : (_temp_var_514 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_513 == 0 ? indices.field_0 : (_temp_var_513 == 1 ? indices.field_1 : (_temp_var_513 == 2 ? indices.field_2 : (_temp_var_513 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_512 == 0 ? indices.field_0 : (_temp_var_512 == 1 ? indices.field_1 : (_temp_var_512 == 2 ? indices.field_2 : (_temp_var_512 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_59(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_62)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_573;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_573 = _block_k_343_(_env_, _kernel_result_62[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_62[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_62[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_62[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_573 = 37;
    }
        
        _result_[_tid_] = temp_stencil_573;
    }
}



// TODO: There should be a better to check if _block_k_345_ is already defined
#ifndef _block_k_345__func
#define _block_k_345__func
__device__ int _block_k_345_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_515 = ((({ int _temp_var_516 = ((({ int _temp_var_517 = ((values[2] % 4));
        (_temp_var_517 == 0 ? indices.field_0 : (_temp_var_517 == 1 ? indices.field_1 : (_temp_var_517 == 2 ? indices.field_2 : (_temp_var_517 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_516 == 0 ? indices.field_0 : (_temp_var_516 == 1 ? indices.field_1 : (_temp_var_516 == 2 ? indices.field_2 : (_temp_var_516 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_515 == 0 ? indices.field_0 : (_temp_var_515 == 1 ? indices.field_1 : (_temp_var_515 == 2 ? indices.field_2 : (_temp_var_515 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_57(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_60)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_574;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_574 = _block_k_345_(_env_, _kernel_result_60[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_60[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_60[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_60[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_574 = 37;
    }
        
        _result_[_tid_] = temp_stencil_574;
    }
}



// TODO: There should be a better to check if _block_k_347_ is already defined
#ifndef _block_k_347__func
#define _block_k_347__func
__device__ int _block_k_347_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_518 = ((({ int _temp_var_519 = ((({ int _temp_var_520 = ((values[2] % 4));
        (_temp_var_520 == 0 ? indices.field_0 : (_temp_var_520 == 1 ? indices.field_1 : (_temp_var_520 == 2 ? indices.field_2 : (_temp_var_520 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_519 == 0 ? indices.field_0 : (_temp_var_519 == 1 ? indices.field_1 : (_temp_var_519 == 2 ? indices.field_2 : (_temp_var_519 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_518 == 0 ? indices.field_0 : (_temp_var_518 == 1 ? indices.field_1 : (_temp_var_518 == 2 ? indices.field_2 : (_temp_var_518 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_55(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_58)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_575;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_575 = _block_k_347_(_env_, _kernel_result_58[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_58[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_58[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_58[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_575 = 37;
    }
        
        _result_[_tid_] = temp_stencil_575;
    }
}



// TODO: There should be a better to check if _block_k_349_ is already defined
#ifndef _block_k_349__func
#define _block_k_349__func
__device__ int _block_k_349_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_521 = ((({ int _temp_var_522 = ((({ int _temp_var_523 = ((values[2] % 4));
        (_temp_var_523 == 0 ? indices.field_0 : (_temp_var_523 == 1 ? indices.field_1 : (_temp_var_523 == 2 ? indices.field_2 : (_temp_var_523 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_522 == 0 ? indices.field_0 : (_temp_var_522 == 1 ? indices.field_1 : (_temp_var_522 == 2 ? indices.field_2 : (_temp_var_522 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_521 == 0 ? indices.field_0 : (_temp_var_521 == 1 ? indices.field_1 : (_temp_var_521 == 2 ? indices.field_2 : (_temp_var_521 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_53(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_56)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_576;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_576 = _block_k_349_(_env_, _kernel_result_56[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_56[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_56[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_56[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_576 = 37;
    }
        
        _result_[_tid_] = temp_stencil_576;
    }
}



// TODO: There should be a better to check if _block_k_351_ is already defined
#ifndef _block_k_351__func
#define _block_k_351__func
__device__ int _block_k_351_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_524 = ((({ int _temp_var_525 = ((({ int _temp_var_526 = ((values[2] % 4));
        (_temp_var_526 == 0 ? indices.field_0 : (_temp_var_526 == 1 ? indices.field_1 : (_temp_var_526 == 2 ? indices.field_2 : (_temp_var_526 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_525 == 0 ? indices.field_0 : (_temp_var_525 == 1 ? indices.field_1 : (_temp_var_525 == 2 ? indices.field_2 : (_temp_var_525 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_524 == 0 ? indices.field_0 : (_temp_var_524 == 1 ? indices.field_1 : (_temp_var_524 == 2 ? indices.field_2 : (_temp_var_524 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_51(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_54)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_577;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_577 = _block_k_351_(_env_, _kernel_result_54[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_54[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_54[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_54[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_577 = 37;
    }
        
        _result_[_tid_] = temp_stencil_577;
    }
}



// TODO: There should be a better to check if _block_k_353_ is already defined
#ifndef _block_k_353__func
#define _block_k_353__func
__device__ int _block_k_353_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_527 = ((({ int _temp_var_528 = ((({ int _temp_var_529 = ((values[2] % 4));
        (_temp_var_529 == 0 ? indices.field_0 : (_temp_var_529 == 1 ? indices.field_1 : (_temp_var_529 == 2 ? indices.field_2 : (_temp_var_529 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_528 == 0 ? indices.field_0 : (_temp_var_528 == 1 ? indices.field_1 : (_temp_var_528 == 2 ? indices.field_2 : (_temp_var_528 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_527 == 0 ? indices.field_0 : (_temp_var_527 == 1 ? indices.field_1 : (_temp_var_527 == 2 ? indices.field_2 : (_temp_var_527 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_49(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_52)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_578;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_578 = _block_k_353_(_env_, _kernel_result_52[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_52[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_52[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_52[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_578 = 37;
    }
        
        _result_[_tid_] = temp_stencil_578;
    }
}



// TODO: There should be a better to check if _block_k_355_ is already defined
#ifndef _block_k_355__func
#define _block_k_355__func
__device__ int _block_k_355_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_530 = ((({ int _temp_var_531 = ((({ int _temp_var_532 = ((values[2] % 4));
        (_temp_var_532 == 0 ? indices.field_0 : (_temp_var_532 == 1 ? indices.field_1 : (_temp_var_532 == 2 ? indices.field_2 : (_temp_var_532 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_531 == 0 ? indices.field_0 : (_temp_var_531 == 1 ? indices.field_1 : (_temp_var_531 == 2 ? indices.field_2 : (_temp_var_531 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_530 == 0 ? indices.field_0 : (_temp_var_530 == 1 ? indices.field_1 : (_temp_var_530 == 2 ? indices.field_2 : (_temp_var_530 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_47(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_50)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_579;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_579 = _block_k_355_(_env_, _kernel_result_50[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_50[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_50[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_50[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_579 = 37;
    }
        
        _result_[_tid_] = temp_stencil_579;
    }
}



// TODO: There should be a better to check if _block_k_357_ is already defined
#ifndef _block_k_357__func
#define _block_k_357__func
__device__ int _block_k_357_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_533 = ((({ int _temp_var_534 = ((({ int _temp_var_535 = ((values[2] % 4));
        (_temp_var_535 == 0 ? indices.field_0 : (_temp_var_535 == 1 ? indices.field_1 : (_temp_var_535 == 2 ? indices.field_2 : (_temp_var_535 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_534 == 0 ? indices.field_0 : (_temp_var_534 == 1 ? indices.field_1 : (_temp_var_534 == 2 ? indices.field_2 : (_temp_var_534 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_533 == 0 ? indices.field_0 : (_temp_var_533 == 1 ? indices.field_1 : (_temp_var_533 == 2 ? indices.field_2 : (_temp_var_533 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_45(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_48)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_580;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_580 = _block_k_357_(_env_, _kernel_result_48[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_48[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_48[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_48[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_580 = 37;
    }
        
        _result_[_tid_] = temp_stencil_580;
    }
}



// TODO: There should be a better to check if _block_k_359_ is already defined
#ifndef _block_k_359__func
#define _block_k_359__func
__device__ int _block_k_359_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_536 = ((({ int _temp_var_537 = ((({ int _temp_var_538 = ((values[2] % 4));
        (_temp_var_538 == 0 ? indices.field_0 : (_temp_var_538 == 1 ? indices.field_1 : (_temp_var_538 == 2 ? indices.field_2 : (_temp_var_538 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_537 == 0 ? indices.field_0 : (_temp_var_537 == 1 ? indices.field_1 : (_temp_var_537 == 2 ? indices.field_2 : (_temp_var_537 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_536 == 0 ? indices.field_0 : (_temp_var_536 == 1 ? indices.field_1 : (_temp_var_536 == 2 ? indices.field_2 : (_temp_var_536 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_43(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_46)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_581;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_581 = _block_k_359_(_env_, _kernel_result_46[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_46[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_46[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_46[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_581 = 37;
    }
        
        _result_[_tid_] = temp_stencil_581;
    }
}



// TODO: There should be a better to check if _block_k_361_ is already defined
#ifndef _block_k_361__func
#define _block_k_361__func
__device__ int _block_k_361_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_539 = ((({ int _temp_var_540 = ((({ int _temp_var_541 = ((values[2] % 4));
        (_temp_var_541 == 0 ? indices.field_0 : (_temp_var_541 == 1 ? indices.field_1 : (_temp_var_541 == 2 ? indices.field_2 : (_temp_var_541 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_540 == 0 ? indices.field_0 : (_temp_var_540 == 1 ? indices.field_1 : (_temp_var_540 == 2 ? indices.field_2 : (_temp_var_540 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_539 == 0 ? indices.field_0 : (_temp_var_539 == 1 ? indices.field_1 : (_temp_var_539 == 2 ? indices.field_2 : (_temp_var_539 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_41(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_44)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_582;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_582 = _block_k_361_(_env_, _kernel_result_44[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_44[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_44[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_44[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_582 = 37;
    }
        
        _result_[_tid_] = temp_stencil_582;
    }
}



// TODO: There should be a better to check if _block_k_363_ is already defined
#ifndef _block_k_363__func
#define _block_k_363__func
__device__ int _block_k_363_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_542 = ((({ int _temp_var_543 = ((({ int _temp_var_544 = ((values[2] % 4));
        (_temp_var_544 == 0 ? indices.field_0 : (_temp_var_544 == 1 ? indices.field_1 : (_temp_var_544 == 2 ? indices.field_2 : (_temp_var_544 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_543 == 0 ? indices.field_0 : (_temp_var_543 == 1 ? indices.field_1 : (_temp_var_543 == 2 ? indices.field_2 : (_temp_var_543 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_542 == 0 ? indices.field_0 : (_temp_var_542 == 1 ? indices.field_1 : (_temp_var_542 == 2 ? indices.field_2 : (_temp_var_542 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_39(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_42)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_583;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_583 = _block_k_363_(_env_, _kernel_result_42[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_42[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_42[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_42[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_583 = 37;
    }
        
        _result_[_tid_] = temp_stencil_583;
    }
}



// TODO: There should be a better to check if _block_k_365_ is already defined
#ifndef _block_k_365__func
#define _block_k_365__func
__device__ int _block_k_365_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_545 = ((({ int _temp_var_546 = ((({ int _temp_var_547 = ((values[2] % 4));
        (_temp_var_547 == 0 ? indices.field_0 : (_temp_var_547 == 1 ? indices.field_1 : (_temp_var_547 == 2 ? indices.field_2 : (_temp_var_547 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_546 == 0 ? indices.field_0 : (_temp_var_546 == 1 ? indices.field_1 : (_temp_var_546 == 2 ? indices.field_2 : (_temp_var_546 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_545 == 0 ? indices.field_0 : (_temp_var_545 == 1 ? indices.field_1 : (_temp_var_545 == 2 ? indices.field_2 : (_temp_var_545 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_37(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_40)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_584;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_584 = _block_k_365_(_env_, _kernel_result_40[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_40[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_40[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_40[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_584 = 37;
    }
        
        _result_[_tid_] = temp_stencil_584;
    }
}



// TODO: There should be a better to check if _block_k_367_ is already defined
#ifndef _block_k_367__func
#define _block_k_367__func
__device__ int _block_k_367_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_548 = ((({ int _temp_var_549 = ((({ int _temp_var_550 = ((values[2] % 4));
        (_temp_var_550 == 0 ? indices.field_0 : (_temp_var_550 == 1 ? indices.field_1 : (_temp_var_550 == 2 ? indices.field_2 : (_temp_var_550 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_549 == 0 ? indices.field_0 : (_temp_var_549 == 1 ? indices.field_1 : (_temp_var_549 == 2 ? indices.field_2 : (_temp_var_549 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_548 == 0 ? indices.field_0 : (_temp_var_548 == 1 ? indices.field_1 : (_temp_var_548 == 2 ? indices.field_2 : (_temp_var_548 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_35(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_38)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_585;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_585 = _block_k_367_(_env_, _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_38[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_585 = 37;
    }
        
        _result_[_tid_] = temp_stencil_585;
    }
}



// TODO: There should be a better to check if _block_k_369_ is already defined
#ifndef _block_k_369__func
#define _block_k_369__func
__device__ int _block_k_369_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_551 = ((({ int _temp_var_552 = ((({ int _temp_var_553 = ((values[2] % 4));
        (_temp_var_553 == 0 ? indices.field_0 : (_temp_var_553 == 1 ? indices.field_1 : (_temp_var_553 == 2 ? indices.field_2 : (_temp_var_553 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_552 == 0 ? indices.field_0 : (_temp_var_552 == 1 ? indices.field_1 : (_temp_var_552 == 2 ? indices.field_2 : (_temp_var_552 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_551 == 0 ? indices.field_0 : (_temp_var_551 == 1 ? indices.field_1 : (_temp_var_551 == 2 ? indices.field_2 : (_temp_var_551 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_33(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_36)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_586;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_586 = _block_k_369_(_env_, _kernel_result_36[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_36[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_36[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_36[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_586 = 37;
    }
        
        _result_[_tid_] = temp_stencil_586;
    }
}



// TODO: There should be a better to check if _block_k_371_ is already defined
#ifndef _block_k_371__func
#define _block_k_371__func
__device__ int _block_k_371_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_554 = ((({ int _temp_var_555 = ((({ int _temp_var_556 = ((values[2] % 4));
        (_temp_var_556 == 0 ? indices.field_0 : (_temp_var_556 == 1 ? indices.field_1 : (_temp_var_556 == 2 ? indices.field_2 : (_temp_var_556 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_555 == 0 ? indices.field_0 : (_temp_var_555 == 1 ? indices.field_1 : (_temp_var_555 == 2 ? indices.field_2 : (_temp_var_555 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_554 == 0 ? indices.field_0 : (_temp_var_554 == 1 ? indices.field_1 : (_temp_var_554 == 2 ? indices.field_2 : (_temp_var_554 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_31(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_34)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_587;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_587 = _block_k_371_(_env_, _kernel_result_34[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_34[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_34[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_34[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_587 = 37;
    }
        
        _result_[_tid_] = temp_stencil_587;
    }
}



// TODO: There should be a better to check if _block_k_373_ is already defined
#ifndef _block_k_373__func
#define _block_k_373__func
__device__ int _block_k_373_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_557 = ((({ int _temp_var_558 = ((({ int _temp_var_559 = ((values[2] % 4));
        (_temp_var_559 == 0 ? indices.field_0 : (_temp_var_559 == 1 ? indices.field_1 : (_temp_var_559 == 2 ? indices.field_2 : (_temp_var_559 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_558 == 0 ? indices.field_0 : (_temp_var_558 == 1 ? indices.field_1 : (_temp_var_558 == 2 ? indices.field_2 : (_temp_var_558 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_557 == 0 ? indices.field_0 : (_temp_var_557 == 1 ? indices.field_1 : (_temp_var_557 == 2 ? indices.field_2 : (_temp_var_557 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_29(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_32)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_588;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_588 = _block_k_373_(_env_, _kernel_result_32[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_32[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_32[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_32[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_588 = 37;
    }
        
        _result_[_tid_] = temp_stencil_588;
    }
}



// TODO: There should be a better to check if _block_k_375_ is already defined
#ifndef _block_k_375__func
#define _block_k_375__func
__device__ int _block_k_375_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_560 = ((({ int _temp_var_561 = ((({ int _temp_var_562 = ((values[2] % 4));
        (_temp_var_562 == 0 ? indices.field_0 : (_temp_var_562 == 1 ? indices.field_1 : (_temp_var_562 == 2 ? indices.field_2 : (_temp_var_562 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_561 == 0 ? indices.field_0 : (_temp_var_561 == 1 ? indices.field_1 : (_temp_var_561 == 2 ? indices.field_2 : (_temp_var_561 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_560 == 0 ? indices.field_0 : (_temp_var_560 == 1 ? indices.field_1 : (_temp_var_560 == 2 ? indices.field_2 : (_temp_var_560 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_27(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_30)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_589;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_589 = _block_k_375_(_env_, _kernel_result_30[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_30[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_30[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_30[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_589 = 37;
    }
        
        _result_[_tid_] = temp_stencil_589;
    }
}



// TODO: There should be a better to check if _block_k_377_ is already defined
#ifndef _block_k_377__func
#define _block_k_377__func
__device__ int _block_k_377_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_563 = ((({ int _temp_var_564 = ((({ int _temp_var_565 = ((values[2] % 4));
        (_temp_var_565 == 0 ? indices.field_0 : (_temp_var_565 == 1 ? indices.field_1 : (_temp_var_565 == 2 ? indices.field_2 : (_temp_var_565 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_564 == 0 ? indices.field_0 : (_temp_var_564 == 1 ? indices.field_1 : (_temp_var_564 == 2 ? indices.field_2 : (_temp_var_564 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_563 == 0 ? indices.field_0 : (_temp_var_563 == 1 ? indices.field_1 : (_temp_var_563 == 2 ? indices.field_2 : (_temp_var_563 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_25(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_28)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_590;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_590 = _block_k_377_(_env_, _kernel_result_28[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_28[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_28[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_28[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_590 = 37;
    }
        
        _result_[_tid_] = temp_stencil_590;
    }
}



// TODO: There should be a better to check if _block_k_379_ is already defined
#ifndef _block_k_379__func
#define _block_k_379__func
__device__ int _block_k_379_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_566 = ((({ int _temp_var_567 = ((({ int _temp_var_568 = ((values[2] % 4));
        (_temp_var_568 == 0 ? indices.field_0 : (_temp_var_568 == 1 ? indices.field_1 : (_temp_var_568 == 2 ? indices.field_2 : (_temp_var_568 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_567 == 0 ? indices.field_0 : (_temp_var_567 == 1 ? indices.field_1 : (_temp_var_567 == 2 ? indices.field_2 : (_temp_var_567 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_566 == 0 ? indices.field_0 : (_temp_var_566 == 1 ? indices.field_1 : (_temp_var_566 == 2 ? indices.field_2 : (_temp_var_566 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_23(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_26)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_591;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_591 = _block_k_379_(_env_, _kernel_result_26[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_26[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_26[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_26[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_591 = 37;
    }
        
        _result_[_tid_] = temp_stencil_591;
    }
}



// TODO: There should be a better to check if _block_k_381_ is already defined
#ifndef _block_k_381__func
#define _block_k_381__func
__device__ int _block_k_381_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_569 = ((({ int _temp_var_570 = ((({ int _temp_var_571 = ((values[2] % 4));
        (_temp_var_571 == 0 ? indices.field_0 : (_temp_var_571 == 1 ? indices.field_1 : (_temp_var_571 == 2 ? indices.field_2 : (_temp_var_571 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_570 == 0 ? indices.field_0 : (_temp_var_570 == 1 ? indices.field_1 : (_temp_var_570 == 2 ? indices.field_2 : (_temp_var_570 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_569 == 0 ? indices.field_0 : (_temp_var_569 == 1 ? indices.field_1 : (_temp_var_569 == 2 ? indices.field_2 : (_temp_var_569 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_21(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_24)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_592;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_592 = _block_k_381_(_env_, _kernel_result_24[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_24[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_24[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_24[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_592 = 37;
    }
        
        _result_[_tid_] = temp_stencil_592;
    }
}



// TODO: There should be a better to check if _block_k_383_ is already defined
#ifndef _block_k_383__func
#define _block_k_383__func
__device__ int _block_k_383_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_572 = ((({ int _temp_var_573 = ((({ int _temp_var_574 = ((values[2] % 4));
        (_temp_var_574 == 0 ? indices.field_0 : (_temp_var_574 == 1 ? indices.field_1 : (_temp_var_574 == 2 ? indices.field_2 : (_temp_var_574 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_573 == 0 ? indices.field_0 : (_temp_var_573 == 1 ? indices.field_1 : (_temp_var_573 == 2 ? indices.field_2 : (_temp_var_573 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_572 == 0 ? indices.field_0 : (_temp_var_572 == 1 ? indices.field_1 : (_temp_var_572 == 2 ? indices.field_2 : (_temp_var_572 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_19(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_22)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_593;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_593 = _block_k_383_(_env_, _kernel_result_22[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_22[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_22[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_22[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_593 = 37;
    }
        
        _result_[_tid_] = temp_stencil_593;
    }
}



// TODO: There should be a better to check if _block_k_385_ is already defined
#ifndef _block_k_385__func
#define _block_k_385__func
__device__ int _block_k_385_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_575 = ((({ int _temp_var_576 = ((({ int _temp_var_577 = ((values[2] % 4));
        (_temp_var_577 == 0 ? indices.field_0 : (_temp_var_577 == 1 ? indices.field_1 : (_temp_var_577 == 2 ? indices.field_2 : (_temp_var_577 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_576 == 0 ? indices.field_0 : (_temp_var_576 == 1 ? indices.field_1 : (_temp_var_576 == 2 ? indices.field_2 : (_temp_var_576 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_575 == 0 ? indices.field_0 : (_temp_var_575 == 1 ? indices.field_1 : (_temp_var_575 == 2 ? indices.field_2 : (_temp_var_575 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_17(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_20)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_594;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_594 = _block_k_385_(_env_, _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_20[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_594 = 37;
    }
        
        _result_[_tid_] = temp_stencil_594;
    }
}



// TODO: There should be a better to check if _block_k_387_ is already defined
#ifndef _block_k_387__func
#define _block_k_387__func
__device__ int _block_k_387_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_578 = ((({ int _temp_var_579 = ((({ int _temp_var_580 = ((values[2] % 4));
        (_temp_var_580 == 0 ? indices.field_0 : (_temp_var_580 == 1 ? indices.field_1 : (_temp_var_580 == 2 ? indices.field_2 : (_temp_var_580 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_579 == 0 ? indices.field_0 : (_temp_var_579 == 1 ? indices.field_1 : (_temp_var_579 == 2 ? indices.field_2 : (_temp_var_579 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_578 == 0 ? indices.field_0 : (_temp_var_578 == 1 ? indices.field_1 : (_temp_var_578 == 2 ? indices.field_2 : (_temp_var_578 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_15(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_18)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_595;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_595 = _block_k_387_(_env_, _kernel_result_18[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_18[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_18[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_18[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_595 = 37;
    }
        
        _result_[_tid_] = temp_stencil_595;
    }
}



// TODO: There should be a better to check if _block_k_389_ is already defined
#ifndef _block_k_389__func
#define _block_k_389__func
__device__ int _block_k_389_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_581 = ((({ int _temp_var_582 = ((({ int _temp_var_583 = ((values[2] % 4));
        (_temp_var_583 == 0 ? indices.field_0 : (_temp_var_583 == 1 ? indices.field_1 : (_temp_var_583 == 2 ? indices.field_2 : (_temp_var_583 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_582 == 0 ? indices.field_0 : (_temp_var_582 == 1 ? indices.field_1 : (_temp_var_582 == 2 ? indices.field_2 : (_temp_var_582 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_581 == 0 ? indices.field_0 : (_temp_var_581 == 1 ? indices.field_1 : (_temp_var_581 == 2 ? indices.field_2 : (_temp_var_581 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_13(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_16)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_596;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_596 = _block_k_389_(_env_, _kernel_result_16[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_16[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_16[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_16[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_596 = 37;
    }
        
        _result_[_tid_] = temp_stencil_596;
    }
}



// TODO: There should be a better to check if _block_k_391_ is already defined
#ifndef _block_k_391__func
#define _block_k_391__func
__device__ int _block_k_391_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_584 = ((({ int _temp_var_585 = ((({ int _temp_var_586 = ((values[2] % 4));
        (_temp_var_586 == 0 ? indices.field_0 : (_temp_var_586 == 1 ? indices.field_1 : (_temp_var_586 == 2 ? indices.field_2 : (_temp_var_586 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_585 == 0 ? indices.field_0 : (_temp_var_585 == 1 ? indices.field_1 : (_temp_var_585 == 2 ? indices.field_2 : (_temp_var_585 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_584 == 0 ? indices.field_0 : (_temp_var_584 == 1 ? indices.field_1 : (_temp_var_584 == 2 ? indices.field_2 : (_temp_var_584 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_11(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_14)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_597;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_597 = _block_k_391_(_env_, _kernel_result_14[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_14[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_14[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_14[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_597 = 37;
    }
        
        _result_[_tid_] = temp_stencil_597;
    }
}



// TODO: There should be a better to check if _block_k_393_ is already defined
#ifndef _block_k_393__func
#define _block_k_393__func
__device__ int _block_k_393_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_587 = ((({ int _temp_var_588 = ((({ int _temp_var_589 = ((values[2] % 4));
        (_temp_var_589 == 0 ? indices.field_0 : (_temp_var_589 == 1 ? indices.field_1 : (_temp_var_589 == 2 ? indices.field_2 : (_temp_var_589 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_588 == 0 ? indices.field_0 : (_temp_var_588 == 1 ? indices.field_1 : (_temp_var_588 == 2 ? indices.field_2 : (_temp_var_588 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_587 == 0 ? indices.field_0 : (_temp_var_587 == 1 ? indices.field_1 : (_temp_var_587 == 2 ? indices.field_2 : (_temp_var_587 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_9(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_12)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_598;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_598 = _block_k_393_(_env_, _kernel_result_12[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_12[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_12[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_12[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_598 = 37;
    }
        
        _result_[_tid_] = temp_stencil_598;
    }
}



// TODO: There should be a better to check if _block_k_395_ is already defined
#ifndef _block_k_395__func
#define _block_k_395__func
__device__ int _block_k_395_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_590 = ((({ int _temp_var_591 = ((({ int _temp_var_592 = ((values[2] % 4));
        (_temp_var_592 == 0 ? indices.field_0 : (_temp_var_592 == 1 ? indices.field_1 : (_temp_var_592 == 2 ? indices.field_2 : (_temp_var_592 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_591 == 0 ? indices.field_0 : (_temp_var_591 == 1 ? indices.field_1 : (_temp_var_591 == 2 ? indices.field_2 : (_temp_var_591 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_590 == 0 ? indices.field_0 : (_temp_var_590 == 1 ? indices.field_1 : (_temp_var_590 == 2 ? indices.field_2 : (_temp_var_590 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_7(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_10)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_599;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_599 = _block_k_395_(_env_, _kernel_result_10[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_10[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_10[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_10[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_599 = 37;
    }
        
        _result_[_tid_] = temp_stencil_599;
    }
}



// TODO: There should be a better to check if _block_k_397_ is already defined
#ifndef _block_k_397__func
#define _block_k_397__func
__device__ int _block_k_397_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_593 = ((({ int _temp_var_594 = ((({ int _temp_var_595 = ((values[2] % 4));
        (_temp_var_595 == 0 ? indices.field_0 : (_temp_var_595 == 1 ? indices.field_1 : (_temp_var_595 == 2 ? indices.field_2 : (_temp_var_595 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_594 == 0 ? indices.field_0 : (_temp_var_594 == 1 ? indices.field_1 : (_temp_var_594 == 2 ? indices.field_2 : (_temp_var_594 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_593 == 0 ? indices.field_0 : (_temp_var_593 == 1 ? indices.field_1 : (_temp_var_593 == 2 ? indices.field_2 : (_temp_var_593 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_8)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_600;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_600 = _block_k_397_(_env_, _kernel_result_8[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_8[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_8[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_8[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_600 = 37;
    }
        
        _result_[_tid_] = temp_stencil_600;
    }
}



// TODO: There should be a better to check if _block_k_399_ is already defined
#ifndef _block_k_399__func
#define _block_k_399__func
__device__ int _block_k_399_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_596 = ((({ int _temp_var_597 = ((({ int _temp_var_598 = ((values[2] % 4));
        (_temp_var_598 == 0 ? indices.field_0 : (_temp_var_598 == 1 ? indices.field_1 : (_temp_var_598 == 2 ? indices.field_2 : (_temp_var_598 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_597 == 0 ? indices.field_0 : (_temp_var_597 == 1 ? indices.field_1 : (_temp_var_597 == 2 ? indices.field_2 : (_temp_var_597 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_596 == 0 ? indices.field_0 : (_temp_var_596 == 1 ? indices.field_1 : (_temp_var_596 == 2 ? indices.field_2 : (_temp_var_596 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_3(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_6)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_601;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_601 = _block_k_399_(_env_, _kernel_result_6[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_6[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_6[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_6[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_601 = 37;
    }
        
        _result_[_tid_] = temp_stencil_601;
    }
}



// TODO: There should be a better to check if _block_k_401_ is already defined
#ifndef _block_k_401__func
#define _block_k_401__func
__device__ int _block_k_401_(environment_t *_env_, int _values_0, int _values_1, int _values_2, int _values_3, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    int values[] = { _values_0, _values_1, _values_2, _values_3 };
    
    {
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_599 = ((({ int _temp_var_600 = ((({ int _temp_var_601 = ((values[2] % 4));
        (_temp_var_601 == 0 ? indices.field_0 : (_temp_var_601 == 1 ? indices.field_1 : (_temp_var_601 == 2 ? indices.field_2 : (_temp_var_601 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_600 == 0 ? indices.field_0 : (_temp_var_600 == 1 ? indices.field_1 : (_temp_var_600 == 2 ? indices.field_2 : (_temp_var_600 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_599 == 0 ? indices.field_0 : (_temp_var_599 == 1 ? indices.field_1 : (_temp_var_599 == 2 ? indices.field_2 : (_temp_var_599 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_4)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_602;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_602 = _block_k_401_(_env_, _kernel_result_4[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_4[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_4[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_4[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_602 = 37;
    }
        
        _result_[_tid_] = temp_stencil_602;
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
    

    /* Launch all kernels */
        timeStartMeasure();
    int * _kernel_result_402;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_402, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_402);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_401<<<39063, 256>>>(dev_env, 10000000, _kernel_result_402);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    int * _kernel_result_400;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_400, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_400);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_399<<<39063, 256>>>(dev_env, 10000000, _kernel_result_400, _kernel_result_402);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_402));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_398;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_398, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_398);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_397<<<39063, 256>>>(dev_env, 10000000, _kernel_result_398, _kernel_result_400);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_400));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_396;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_396, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_396);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_395<<<39063, 256>>>(dev_env, 10000000, _kernel_result_396, _kernel_result_398);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_398));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_394;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_394, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_394);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_393<<<39063, 256>>>(dev_env, 10000000, _kernel_result_394, _kernel_result_396);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_396));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_392;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_392, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_392);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_391<<<39063, 256>>>(dev_env, 10000000, _kernel_result_392, _kernel_result_394);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_394));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_390;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_390, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_390);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_389<<<39063, 256>>>(dev_env, 10000000, _kernel_result_390, _kernel_result_392);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_392));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_388;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_388, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_388);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_387<<<39063, 256>>>(dev_env, 10000000, _kernel_result_388, _kernel_result_390);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_390));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_386;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_386, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_386);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_385<<<39063, 256>>>(dev_env, 10000000, _kernel_result_386, _kernel_result_388);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_388));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_384;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_384, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_384);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_383<<<39063, 256>>>(dev_env, 10000000, _kernel_result_384, _kernel_result_386);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_386));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_382;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_382, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_382);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_381<<<39063, 256>>>(dev_env, 10000000, _kernel_result_382, _kernel_result_384);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_384));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_380;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_380, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_380);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_379<<<39063, 256>>>(dev_env, 10000000, _kernel_result_380, _kernel_result_382);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_382));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_378;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_378, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_378);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_377<<<39063, 256>>>(dev_env, 10000000, _kernel_result_378, _kernel_result_380);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_380));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_376;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_376, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_376);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_375<<<39063, 256>>>(dev_env, 10000000, _kernel_result_376, _kernel_result_378);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_378));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_374;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_374, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_374);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_373<<<39063, 256>>>(dev_env, 10000000, _kernel_result_374, _kernel_result_376);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_376));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_372;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_372, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_372);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_371<<<39063, 256>>>(dev_env, 10000000, _kernel_result_372, _kernel_result_374);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_374));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_370;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_370, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_370);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_369<<<39063, 256>>>(dev_env, 10000000, _kernel_result_370, _kernel_result_372);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_372));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_368;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_368, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_368);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_367<<<39063, 256>>>(dev_env, 10000000, _kernel_result_368, _kernel_result_370);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_370));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_366;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_366, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_366);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_365<<<39063, 256>>>(dev_env, 10000000, _kernel_result_366, _kernel_result_368);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_368));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_364;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_364, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_364);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_363<<<39063, 256>>>(dev_env, 10000000, _kernel_result_364, _kernel_result_366);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_366));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_362;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_362, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_362);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_361<<<39063, 256>>>(dev_env, 10000000, _kernel_result_362, _kernel_result_364);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_364));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_360;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_360, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_360);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_359<<<39063, 256>>>(dev_env, 10000000, _kernel_result_360, _kernel_result_362);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_362));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_358;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_358, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_358);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_357<<<39063, 256>>>(dev_env, 10000000, _kernel_result_358, _kernel_result_360);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_360));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_356;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_356, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_356);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_355<<<39063, 256>>>(dev_env, 10000000, _kernel_result_356, _kernel_result_358);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_358));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_354;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_354, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_354);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_353<<<39063, 256>>>(dev_env, 10000000, _kernel_result_354, _kernel_result_356);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_356));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_352;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_352, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_352);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_351<<<39063, 256>>>(dev_env, 10000000, _kernel_result_352, _kernel_result_354);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_354));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_350;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_350, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_350);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_349<<<39063, 256>>>(dev_env, 10000000, _kernel_result_350, _kernel_result_352);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_352));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_348;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_348, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_348);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_347<<<39063, 256>>>(dev_env, 10000000, _kernel_result_348, _kernel_result_350);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_350));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_346;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_346, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_346);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_345<<<39063, 256>>>(dev_env, 10000000, _kernel_result_346, _kernel_result_348);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_348));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_344;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_344, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_344);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_343<<<39063, 256>>>(dev_env, 10000000, _kernel_result_344, _kernel_result_346);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_346));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_342;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_342, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_342);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_341<<<39063, 256>>>(dev_env, 10000000, _kernel_result_342, _kernel_result_344);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_344));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_340;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_340, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_340);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_339<<<39063, 256>>>(dev_env, 10000000, _kernel_result_340, _kernel_result_342);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_342));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_338;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_338, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_338);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_337<<<39063, 256>>>(dev_env, 10000000, _kernel_result_338, _kernel_result_340);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_340));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_336;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_336, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_336);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_335<<<39063, 256>>>(dev_env, 10000000, _kernel_result_336, _kernel_result_338);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_338));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_334;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_334, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_334);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_333<<<39063, 256>>>(dev_env, 10000000, _kernel_result_334, _kernel_result_336);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_336));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_332;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_332, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_332);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_331<<<39063, 256>>>(dev_env, 10000000, _kernel_result_332, _kernel_result_334);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_334));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_330;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_330, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_330);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_329<<<39063, 256>>>(dev_env, 10000000, _kernel_result_330, _kernel_result_332);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_332));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_328;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_328, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_328);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_327<<<39063, 256>>>(dev_env, 10000000, _kernel_result_328, _kernel_result_330);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_330));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_326;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_326, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_326);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_325<<<39063, 256>>>(dev_env, 10000000, _kernel_result_326, _kernel_result_328);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_328));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_324;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_324, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_324);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_323<<<39063, 256>>>(dev_env, 10000000, _kernel_result_324, _kernel_result_326);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_326));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_322;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_322, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_322);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_321<<<39063, 256>>>(dev_env, 10000000, _kernel_result_322, _kernel_result_324);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_324));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_320;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_320, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_320);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_319<<<39063, 256>>>(dev_env, 10000000, _kernel_result_320, _kernel_result_322);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_322));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_318;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_318, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_318);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_317<<<39063, 256>>>(dev_env, 10000000, _kernel_result_318, _kernel_result_320);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_320));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_316;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_316, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_316);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_315<<<39063, 256>>>(dev_env, 10000000, _kernel_result_316, _kernel_result_318);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_318));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_314;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_314, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_314);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_313<<<39063, 256>>>(dev_env, 10000000, _kernel_result_314, _kernel_result_316);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_316));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_312;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_312, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_312);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_311<<<39063, 256>>>(dev_env, 10000000, _kernel_result_312, _kernel_result_314);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_314));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_310;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_310, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_310);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_309<<<39063, 256>>>(dev_env, 10000000, _kernel_result_310, _kernel_result_312);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_312));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_308;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_308, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_308);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_307<<<39063, 256>>>(dev_env, 10000000, _kernel_result_308, _kernel_result_310);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_310));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_306;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_306, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_306);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_305<<<39063, 256>>>(dev_env, 10000000, _kernel_result_306, _kernel_result_308);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_308));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_304;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_304, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_304);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_303<<<39063, 256>>>(dev_env, 10000000, _kernel_result_304, _kernel_result_306);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_306));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_302;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_302, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_302);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_301<<<39063, 256>>>(dev_env, 10000000, _kernel_result_302, _kernel_result_304);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_304));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_300;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_300, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_300);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_299<<<39063, 256>>>(dev_env, 10000000, _kernel_result_300, _kernel_result_302);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_302));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_298;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_298, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_298);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_297<<<39063, 256>>>(dev_env, 10000000, _kernel_result_298, _kernel_result_300);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_300));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_296;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_296, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_296);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_295<<<39063, 256>>>(dev_env, 10000000, _kernel_result_296, _kernel_result_298);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_298));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_294;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_294, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_294);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_293<<<39063, 256>>>(dev_env, 10000000, _kernel_result_294, _kernel_result_296);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_296));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_292;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_292, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_292);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_291<<<39063, 256>>>(dev_env, 10000000, _kernel_result_292, _kernel_result_294);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_294));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_290;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_290, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_290);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_289<<<39063, 256>>>(dev_env, 10000000, _kernel_result_290, _kernel_result_292);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_292));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_288;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_288, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_288);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_287<<<39063, 256>>>(dev_env, 10000000, _kernel_result_288, _kernel_result_290);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_290));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_286;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_286, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_286);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_285<<<39063, 256>>>(dev_env, 10000000, _kernel_result_286, _kernel_result_288);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_288));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_284;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_284, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_284);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_283<<<39063, 256>>>(dev_env, 10000000, _kernel_result_284, _kernel_result_286);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_286));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_282;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_282, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_282);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_281<<<39063, 256>>>(dev_env, 10000000, _kernel_result_282, _kernel_result_284);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_284));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_280;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_280, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_280);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_279<<<39063, 256>>>(dev_env, 10000000, _kernel_result_280, _kernel_result_282);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_282));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_278;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_278, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_278);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_277<<<39063, 256>>>(dev_env, 10000000, _kernel_result_278, _kernel_result_280);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_280));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_276;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_276, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_276);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_275<<<39063, 256>>>(dev_env, 10000000, _kernel_result_276, _kernel_result_278);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_278));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_274;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_274, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_274);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_273<<<39063, 256>>>(dev_env, 10000000, _kernel_result_274, _kernel_result_276);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_276));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_272;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_272, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_272);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_271<<<39063, 256>>>(dev_env, 10000000, _kernel_result_272, _kernel_result_274);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_274));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_270;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_270, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_270);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_269<<<39063, 256>>>(dev_env, 10000000, _kernel_result_270, _kernel_result_272);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_272));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_268;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_268, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_268);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_267<<<39063, 256>>>(dev_env, 10000000, _kernel_result_268, _kernel_result_270);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_270));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_266;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_266, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_266);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_265<<<39063, 256>>>(dev_env, 10000000, _kernel_result_266, _kernel_result_268);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_268));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_264;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_264, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_264);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_263<<<39063, 256>>>(dev_env, 10000000, _kernel_result_264, _kernel_result_266);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_266));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_262;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_262, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_262);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_261<<<39063, 256>>>(dev_env, 10000000, _kernel_result_262, _kernel_result_264);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_264));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_260;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_260, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_260);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_259<<<39063, 256>>>(dev_env, 10000000, _kernel_result_260, _kernel_result_262);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_262));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_258;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_258, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_258);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_257<<<39063, 256>>>(dev_env, 10000000, _kernel_result_258, _kernel_result_260);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_260));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_256;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_256, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_256);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_255<<<39063, 256>>>(dev_env, 10000000, _kernel_result_256, _kernel_result_258);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_258));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_254;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_254, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_254);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_253<<<39063, 256>>>(dev_env, 10000000, _kernel_result_254, _kernel_result_256);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_256));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_252;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_252, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_252);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_251<<<39063, 256>>>(dev_env, 10000000, _kernel_result_252, _kernel_result_254);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_254));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_250;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_250, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_250);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_249<<<39063, 256>>>(dev_env, 10000000, _kernel_result_250, _kernel_result_252);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_252));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_248;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_248, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_248);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_247<<<39063, 256>>>(dev_env, 10000000, _kernel_result_248, _kernel_result_250);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_250));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_246;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_246, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_246);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_245<<<39063, 256>>>(dev_env, 10000000, _kernel_result_246, _kernel_result_248);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_248));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_244;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_244, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_244);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_243<<<39063, 256>>>(dev_env, 10000000, _kernel_result_244, _kernel_result_246);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_246));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_242;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_242, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_242);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_241<<<39063, 256>>>(dev_env, 10000000, _kernel_result_242, _kernel_result_244);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_244));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_240;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_240, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_240);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_239<<<39063, 256>>>(dev_env, 10000000, _kernel_result_240, _kernel_result_242);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_242));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_238;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_238, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_238);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_237<<<39063, 256>>>(dev_env, 10000000, _kernel_result_238, _kernel_result_240);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_240));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_236;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_236, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_236);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_235<<<39063, 256>>>(dev_env, 10000000, _kernel_result_236, _kernel_result_238);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_238));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_234;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_234, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_234);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_233<<<39063, 256>>>(dev_env, 10000000, _kernel_result_234, _kernel_result_236);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_236));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_232;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_232, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_232);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_231<<<39063, 256>>>(dev_env, 10000000, _kernel_result_232, _kernel_result_234);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_234));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_230;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_230, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_230);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_229<<<39063, 256>>>(dev_env, 10000000, _kernel_result_230, _kernel_result_232);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_232));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_228;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_228, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_228);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_227<<<39063, 256>>>(dev_env, 10000000, _kernel_result_228, _kernel_result_230);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_230));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_226;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_226, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_226);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_225<<<39063, 256>>>(dev_env, 10000000, _kernel_result_226, _kernel_result_228);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_228));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_224;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_224, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_224);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_223<<<39063, 256>>>(dev_env, 10000000, _kernel_result_224, _kernel_result_226);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_226));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_222;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_222, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_222);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_221<<<39063, 256>>>(dev_env, 10000000, _kernel_result_222, _kernel_result_224);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_224));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_220;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_220, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_220);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_219<<<39063, 256>>>(dev_env, 10000000, _kernel_result_220, _kernel_result_222);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_222));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_218;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_218, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_218);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_217<<<39063, 256>>>(dev_env, 10000000, _kernel_result_218, _kernel_result_220);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_220));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_216;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_216, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_216);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_215<<<39063, 256>>>(dev_env, 10000000, _kernel_result_216, _kernel_result_218);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_218));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_214;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_214, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_214);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_213<<<39063, 256>>>(dev_env, 10000000, _kernel_result_214, _kernel_result_216);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_216));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_212;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_212, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_212);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_211<<<39063, 256>>>(dev_env, 10000000, _kernel_result_212, _kernel_result_214);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_214));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_210;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_210, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_210);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_209<<<39063, 256>>>(dev_env, 10000000, _kernel_result_210, _kernel_result_212);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_212));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_208;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_208, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_208);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_207<<<39063, 256>>>(dev_env, 10000000, _kernel_result_208, _kernel_result_210);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_210));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_206;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_206, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_206);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_205<<<39063, 256>>>(dev_env, 10000000, _kernel_result_206, _kernel_result_208);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_208));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_204;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_204, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_204);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_203<<<39063, 256>>>(dev_env, 10000000, _kernel_result_204, _kernel_result_206);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_206));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_202;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_202, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_202);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_201<<<39063, 256>>>(dev_env, 10000000, _kernel_result_202, _kernel_result_204);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_204));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_200;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_200, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_200);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_199<<<39063, 256>>>(dev_env, 10000000, _kernel_result_200, _kernel_result_202);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_202));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_198;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_198, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_198);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_197<<<39063, 256>>>(dev_env, 10000000, _kernel_result_198, _kernel_result_200);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_200));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_196;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_196, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_196);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_195<<<39063, 256>>>(dev_env, 10000000, _kernel_result_196, _kernel_result_198);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_198));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_194;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_194, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_194);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_193<<<39063, 256>>>(dev_env, 10000000, _kernel_result_194, _kernel_result_196);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_196));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_192;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_192, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_192);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_191<<<39063, 256>>>(dev_env, 10000000, _kernel_result_192, _kernel_result_194);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_194));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_190;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_190, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_190);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_189<<<39063, 256>>>(dev_env, 10000000, _kernel_result_190, _kernel_result_192);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_192));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_188;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_188, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_188);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_187<<<39063, 256>>>(dev_env, 10000000, _kernel_result_188, _kernel_result_190);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_190));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_186;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_186, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_186);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_185<<<39063, 256>>>(dev_env, 10000000, _kernel_result_186, _kernel_result_188);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_188));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_184;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_184, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_184);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_183<<<39063, 256>>>(dev_env, 10000000, _kernel_result_184, _kernel_result_186);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_186));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_182;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_182, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_182);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_181<<<39063, 256>>>(dev_env, 10000000, _kernel_result_182, _kernel_result_184);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_184));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_180;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_180, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_180);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_179<<<39063, 256>>>(dev_env, 10000000, _kernel_result_180, _kernel_result_182);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_182));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_178;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_178, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_178);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_177<<<39063, 256>>>(dev_env, 10000000, _kernel_result_178, _kernel_result_180);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_180));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_176;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_176, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_176);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_175<<<39063, 256>>>(dev_env, 10000000, _kernel_result_176, _kernel_result_178);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_178));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_174;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_174, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_174);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_173<<<39063, 256>>>(dev_env, 10000000, _kernel_result_174, _kernel_result_176);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_176));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_172;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_172, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_172);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_171<<<39063, 256>>>(dev_env, 10000000, _kernel_result_172, _kernel_result_174);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_174));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_170;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_170, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_170);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_169<<<39063, 256>>>(dev_env, 10000000, _kernel_result_170, _kernel_result_172);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_172));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_168;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_168, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_168);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_167<<<39063, 256>>>(dev_env, 10000000, _kernel_result_168, _kernel_result_170);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_170));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_166;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_166, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_166);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_165<<<39063, 256>>>(dev_env, 10000000, _kernel_result_166, _kernel_result_168);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_168));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_164;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_164, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_164);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_163<<<39063, 256>>>(dev_env, 10000000, _kernel_result_164, _kernel_result_166);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_166));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_162;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_162, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_162);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_161<<<39063, 256>>>(dev_env, 10000000, _kernel_result_162, _kernel_result_164);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_164));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_160;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_160, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_160);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_159<<<39063, 256>>>(dev_env, 10000000, _kernel_result_160, _kernel_result_162);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_162));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_158;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_158, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_158);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_157<<<39063, 256>>>(dev_env, 10000000, _kernel_result_158, _kernel_result_160);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_160));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_156;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_156, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_156);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_155<<<39063, 256>>>(dev_env, 10000000, _kernel_result_156, _kernel_result_158);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_158));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_154;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_154, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_154);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_153<<<39063, 256>>>(dev_env, 10000000, _kernel_result_154, _kernel_result_156);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_156));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_152;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_152, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_152);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_151<<<39063, 256>>>(dev_env, 10000000, _kernel_result_152, _kernel_result_154);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_154));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_150;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_150, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_150);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_149<<<39063, 256>>>(dev_env, 10000000, _kernel_result_150, _kernel_result_152);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_152));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_148;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_148, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_148);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_147<<<39063, 256>>>(dev_env, 10000000, _kernel_result_148, _kernel_result_150);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_150));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_146;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_146, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_146);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_145<<<39063, 256>>>(dev_env, 10000000, _kernel_result_146, _kernel_result_148);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_148));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_144;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_144, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_144);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_143<<<39063, 256>>>(dev_env, 10000000, _kernel_result_144, _kernel_result_146);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_146));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_142;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_142, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_142);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_141<<<39063, 256>>>(dev_env, 10000000, _kernel_result_142, _kernel_result_144);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_144));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_140;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_140, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_140);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_139<<<39063, 256>>>(dev_env, 10000000, _kernel_result_140, _kernel_result_142);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_142));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_138;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_138, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_138);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_137<<<39063, 256>>>(dev_env, 10000000, _kernel_result_138, _kernel_result_140);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_140));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_136;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_136, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_136);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_135<<<39063, 256>>>(dev_env, 10000000, _kernel_result_136, _kernel_result_138);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_138));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_134;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_134, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_134);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_133<<<39063, 256>>>(dev_env, 10000000, _kernel_result_134, _kernel_result_136);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_136));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_132;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_132, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_132);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_131<<<39063, 256>>>(dev_env, 10000000, _kernel_result_132, _kernel_result_134);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_134));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_130;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_130, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_130);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_129<<<39063, 256>>>(dev_env, 10000000, _kernel_result_130, _kernel_result_132);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_132));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_128;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_128, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_128);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_127<<<39063, 256>>>(dev_env, 10000000, _kernel_result_128, _kernel_result_130);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_130));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_126;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_126, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_126);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_125<<<39063, 256>>>(dev_env, 10000000, _kernel_result_126, _kernel_result_128);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_128));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_124;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_124, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_124);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_123<<<39063, 256>>>(dev_env, 10000000, _kernel_result_124, _kernel_result_126);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_126));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_122;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_122, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_122);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_121<<<39063, 256>>>(dev_env, 10000000, _kernel_result_122, _kernel_result_124);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_124));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_120;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_120, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_120);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_119<<<39063, 256>>>(dev_env, 10000000, _kernel_result_120, _kernel_result_122);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_122));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_118;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_118, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_118);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_117<<<39063, 256>>>(dev_env, 10000000, _kernel_result_118, _kernel_result_120);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_120));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_116;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_116, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_116);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_115<<<39063, 256>>>(dev_env, 10000000, _kernel_result_116, _kernel_result_118);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_118));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_114;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_114, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_114);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_113<<<39063, 256>>>(dev_env, 10000000, _kernel_result_114, _kernel_result_116);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_116));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_112;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_112, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_112);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_111<<<39063, 256>>>(dev_env, 10000000, _kernel_result_112, _kernel_result_114);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_114));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_110;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_110, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_110);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_109<<<39063, 256>>>(dev_env, 10000000, _kernel_result_110, _kernel_result_112);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_112));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_108;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_108, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_108);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_107<<<39063, 256>>>(dev_env, 10000000, _kernel_result_108, _kernel_result_110);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_110));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_106;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_106, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_106);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_105<<<39063, 256>>>(dev_env, 10000000, _kernel_result_106, _kernel_result_108);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_108));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_104;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_104, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_104);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_103<<<39063, 256>>>(dev_env, 10000000, _kernel_result_104, _kernel_result_106);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_106));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_102;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_102, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_102);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_101<<<39063, 256>>>(dev_env, 10000000, _kernel_result_102, _kernel_result_104);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_104));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_100;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_100, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_100);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_99<<<39063, 256>>>(dev_env, 10000000, _kernel_result_100, _kernel_result_102);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_102));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_98;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_98, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_98);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_97<<<39063, 256>>>(dev_env, 10000000, _kernel_result_98, _kernel_result_100);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_100));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_96;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_96, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_96);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_95<<<39063, 256>>>(dev_env, 10000000, _kernel_result_96, _kernel_result_98);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_98));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_94;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_94, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_94);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_93<<<39063, 256>>>(dev_env, 10000000, _kernel_result_94, _kernel_result_96);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_96));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_92;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_92, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_92);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_91<<<39063, 256>>>(dev_env, 10000000, _kernel_result_92, _kernel_result_94);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_94));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_90;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_90, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_90);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_89<<<39063, 256>>>(dev_env, 10000000, _kernel_result_90, _kernel_result_92);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_92));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_88;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_88, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_88);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_87<<<39063, 256>>>(dev_env, 10000000, _kernel_result_88, _kernel_result_90);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_90));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_86;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_86, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_86);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_85<<<39063, 256>>>(dev_env, 10000000, _kernel_result_86, _kernel_result_88);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_88));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_84;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_84, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_84);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_83<<<39063, 256>>>(dev_env, 10000000, _kernel_result_84, _kernel_result_86);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_86));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_82;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_82, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_82);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_81<<<39063, 256>>>(dev_env, 10000000, _kernel_result_82, _kernel_result_84);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_84));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_80;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_80, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_80);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_79<<<39063, 256>>>(dev_env, 10000000, _kernel_result_80, _kernel_result_82);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_82));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_78;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_78, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_78);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_77<<<39063, 256>>>(dev_env, 10000000, _kernel_result_78, _kernel_result_80);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_80));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_76;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_76, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_76);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_75<<<39063, 256>>>(dev_env, 10000000, _kernel_result_76, _kernel_result_78);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_78));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_74;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_74, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_74);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_73<<<39063, 256>>>(dev_env, 10000000, _kernel_result_74, _kernel_result_76);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_76));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_72;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_72, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_72);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_71<<<39063, 256>>>(dev_env, 10000000, _kernel_result_72, _kernel_result_74);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_74));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_70;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_70, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_70);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_69<<<39063, 256>>>(dev_env, 10000000, _kernel_result_70, _kernel_result_72);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_72));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_68;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_68, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_68);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_67<<<39063, 256>>>(dev_env, 10000000, _kernel_result_68, _kernel_result_70);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_70));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_66;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_66, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_66);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_65<<<39063, 256>>>(dev_env, 10000000, _kernel_result_66, _kernel_result_68);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_68));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_64;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_64, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_64);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_63<<<39063, 256>>>(dev_env, 10000000, _kernel_result_64, _kernel_result_66);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_66));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_62;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_62, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_62);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_61<<<39063, 256>>>(dev_env, 10000000, _kernel_result_62, _kernel_result_64);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_64));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_60;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_60, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_60);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_59<<<39063, 256>>>(dev_env, 10000000, _kernel_result_60, _kernel_result_62);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_62));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_58;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_58, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_58);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_57<<<39063, 256>>>(dev_env, 10000000, _kernel_result_58, _kernel_result_60);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_60));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_56;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_56, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_56);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_55<<<39063, 256>>>(dev_env, 10000000, _kernel_result_56, _kernel_result_58);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_58));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_54;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_54, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_54);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_53<<<39063, 256>>>(dev_env, 10000000, _kernel_result_54, _kernel_result_56);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_56));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_52;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_52, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_52);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_51<<<39063, 256>>>(dev_env, 10000000, _kernel_result_52, _kernel_result_54);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_54));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_50;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_50, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_50);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_49<<<39063, 256>>>(dev_env, 10000000, _kernel_result_50, _kernel_result_52);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_52));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_48;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_48, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_48);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_47<<<39063, 256>>>(dev_env, 10000000, _kernel_result_48, _kernel_result_50);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_50));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_46;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_46, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_46);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_45<<<39063, 256>>>(dev_env, 10000000, _kernel_result_46, _kernel_result_48);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_48));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_44;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_44, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_44);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_43<<<39063, 256>>>(dev_env, 10000000, _kernel_result_44, _kernel_result_46);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_46));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_42;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_42, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_42);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_41<<<39063, 256>>>(dev_env, 10000000, _kernel_result_42, _kernel_result_44);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_44));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_40;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_40, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_40);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_39<<<39063, 256>>>(dev_env, 10000000, _kernel_result_40, _kernel_result_42);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_42));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_38;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_38, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_38);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_37<<<39063, 256>>>(dev_env, 10000000, _kernel_result_38, _kernel_result_40);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_40));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_36;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_36, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_36);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_35<<<39063, 256>>>(dev_env, 10000000, _kernel_result_36, _kernel_result_38);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_38));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_34;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_34, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_34);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_33<<<39063, 256>>>(dev_env, 10000000, _kernel_result_34, _kernel_result_36);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_36));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_32;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_32, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_32);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_31<<<39063, 256>>>(dev_env, 10000000, _kernel_result_32, _kernel_result_34);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_34));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_30;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_30, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_30);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_29<<<39063, 256>>>(dev_env, 10000000, _kernel_result_30, _kernel_result_32);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_32));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_28;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_28, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_28);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_27<<<39063, 256>>>(dev_env, 10000000, _kernel_result_28, _kernel_result_30);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_30));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_26;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_26, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_26);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_25<<<39063, 256>>>(dev_env, 10000000, _kernel_result_26, _kernel_result_28);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_28));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_24;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_24, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_24);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_23<<<39063, 256>>>(dev_env, 10000000, _kernel_result_24, _kernel_result_26);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_26));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_22;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_22, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_22);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_21<<<39063, 256>>>(dev_env, 10000000, _kernel_result_22, _kernel_result_24);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_24));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_20;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_20, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_20);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_19<<<39063, 256>>>(dev_env, 10000000, _kernel_result_20, _kernel_result_22);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_22));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_18;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_18, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_18);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_17<<<39063, 256>>>(dev_env, 10000000, _kernel_result_18, _kernel_result_20);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_20));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_16;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_16, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_16);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_15<<<39063, 256>>>(dev_env, 10000000, _kernel_result_16, _kernel_result_18);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_18));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_14;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_14, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_14);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_13<<<39063, 256>>>(dev_env, 10000000, _kernel_result_14, _kernel_result_16);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_16));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_12;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_12, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_12);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_11<<<39063, 256>>>(dev_env, 10000000, _kernel_result_12, _kernel_result_14);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_14));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_10;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_10, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_10);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_9<<<39063, 256>>>(dev_env, 10000000, _kernel_result_10, _kernel_result_12);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_12));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_8;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_8, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_8);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_7<<<39063, 256>>>(dev_env, 10000000, _kernel_result_8, _kernel_result_10);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_10));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_6;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_6);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_5<<<39063, 256>>>(dev_env, 10000000, _kernel_result_6, _kernel_result_8);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_8));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_4;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_4, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_4);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_3<<<39063, 256>>>(dev_env, 10000000, _kernel_result_4, _kernel_result_6);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_6));
    timeReportMeasure(program_result, free_memory);
    timeStartMeasure();
    int * _kernel_result_2;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_2);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_1<<<39063, 256>>>(dev_env, 10000000, _kernel_result_2, _kernel_result_4);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_4));
    timeReportMeasure(program_result, free_memory);


    /* Copy over result to the host */
    program_result->result = ({
    variable_size_array_t device_array = variable_size_array_t((void *) _kernel_result_2, 10000000);
    int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
    timeReportMeasure(program_result, transfer_memory);

    variable_size_array_t((void *) tmp_result, device_array.size);
});

    /* Free device memory */
    

    delete program_result->device_allocations;
    
    return program_result;
}
