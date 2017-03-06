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
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_237 = ((indices.field_1 % 4));
        (_temp_var_237 == 0 ? indices.field_0 : (_temp_var_237 == 1 ? indices.field_1 : (_temp_var_237 == 2 ? indices.field_2 : (_temp_var_237 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_473(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    {
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_238 = ((indices.field_1 % 4));
        (_temp_var_238 == 0 ? indices.field_0 : (_temp_var_238 == 1 ? indices.field_1 : (_temp_var_238 == 2 ? indices.field_2 : (_temp_var_238 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_475(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_479(environment_t *_env_, int _num_threads_, int *_result_, int *_array_481_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_481_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_239 = ((({ int _temp_var_240 = ((({ int _temp_var_241 = ((values[2] % 4));
        (_temp_var_241 == 0 ? indices.field_0 : (_temp_var_241 == 1 ? indices.field_1 : (_temp_var_241 == 2 ? indices.field_2 : (_temp_var_241 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_240 == 0 ? indices.field_0 : (_temp_var_240 == 1 ? indices.field_1 : (_temp_var_240 == 2 ? indices.field_2 : (_temp_var_240 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_239 == 0 ? indices.field_0 : (_temp_var_239 == 1 ? indices.field_1 : (_temp_var_239 == 2 ? indices.field_2 : (_temp_var_239 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_477(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_480)
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
        
        temp_stencil_482 = _block_k_4_(_env_, _kernel_result_480[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_480[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_480[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_480[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_482 = 37;
    }
        
        _result_[_tid_] = temp_stencil_482;
    }
}




__global__ void kernel_485(environment_t *_env_, int _num_threads_, int *_result_, int *_array_487_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_487_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_242 = ((({ int _temp_var_243 = ((({ int _temp_var_244 = ((values[2] % 4));
        (_temp_var_244 == 0 ? indices.field_0 : (_temp_var_244 == 1 ? indices.field_1 : (_temp_var_244 == 2 ? indices.field_2 : (_temp_var_244 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_243 == 0 ? indices.field_0 : (_temp_var_243 == 1 ? indices.field_1 : (_temp_var_243 == 2 ? indices.field_2 : (_temp_var_243 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_242 == 0 ? indices.field_0 : (_temp_var_242 == 1 ? indices.field_1 : (_temp_var_242 == 2 ? indices.field_2 : (_temp_var_242 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_483(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_486)
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
        
        temp_stencil_488 = _block_k_4_(_env_, _kernel_result_486[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_486[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_486[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_486[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_488 = 37;
    }
        
        _result_[_tid_] = temp_stencil_488;
    }
}




__global__ void kernel_491(environment_t *_env_, int _num_threads_, int *_result_, int *_array_493_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_493_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_245 = ((({ int _temp_var_246 = ((({ int _temp_var_247 = ((values[2] % 4));
        (_temp_var_247 == 0 ? indices.field_0 : (_temp_var_247 == 1 ? indices.field_1 : (_temp_var_247 == 2 ? indices.field_2 : (_temp_var_247 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_246 == 0 ? indices.field_0 : (_temp_var_246 == 1 ? indices.field_1 : (_temp_var_246 == 2 ? indices.field_2 : (_temp_var_246 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_245 == 0 ? indices.field_0 : (_temp_var_245 == 1 ? indices.field_1 : (_temp_var_245 == 2 ? indices.field_2 : (_temp_var_245 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_489(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_492)
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
        
        temp_stencil_494 = _block_k_4_(_env_, _kernel_result_492[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_492[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_492[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_492[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_494 = 37;
    }
        
        _result_[_tid_] = temp_stencil_494;
    }
}




__global__ void kernel_497(environment_t *_env_, int _num_threads_, int *_result_, int *_array_499_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_499_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_248 = ((({ int _temp_var_249 = ((({ int _temp_var_250 = ((values[2] % 4));
        (_temp_var_250 == 0 ? indices.field_0 : (_temp_var_250 == 1 ? indices.field_1 : (_temp_var_250 == 2 ? indices.field_2 : (_temp_var_250 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_249 == 0 ? indices.field_0 : (_temp_var_249 == 1 ? indices.field_1 : (_temp_var_249 == 2 ? indices.field_2 : (_temp_var_249 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_248 == 0 ? indices.field_0 : (_temp_var_248 == 1 ? indices.field_1 : (_temp_var_248 == 2 ? indices.field_2 : (_temp_var_248 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_495(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_498)
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
        
        temp_stencil_500 = _block_k_4_(_env_, _kernel_result_498[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_498[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_498[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_498[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_500 = 37;
    }
        
        _result_[_tid_] = temp_stencil_500;
    }
}




__global__ void kernel_503(environment_t *_env_, int _num_threads_, int *_result_, int *_array_505_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_505_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_251 = ((({ int _temp_var_252 = ((({ int _temp_var_253 = ((values[2] % 4));
        (_temp_var_253 == 0 ? indices.field_0 : (_temp_var_253 == 1 ? indices.field_1 : (_temp_var_253 == 2 ? indices.field_2 : (_temp_var_253 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_252 == 0 ? indices.field_0 : (_temp_var_252 == 1 ? indices.field_1 : (_temp_var_252 == 2 ? indices.field_2 : (_temp_var_252 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_251 == 0 ? indices.field_0 : (_temp_var_251 == 1 ? indices.field_1 : (_temp_var_251 == 2 ? indices.field_2 : (_temp_var_251 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_501(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_504)
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
        
        temp_stencil_506 = _block_k_4_(_env_, _kernel_result_504[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_504[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_504[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_504[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_506 = 37;
    }
        
        _result_[_tid_] = temp_stencil_506;
    }
}




__global__ void kernel_509(environment_t *_env_, int _num_threads_, int *_result_, int *_array_511_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_511_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_254 = ((({ int _temp_var_255 = ((({ int _temp_var_256 = ((values[2] % 4));
        (_temp_var_256 == 0 ? indices.field_0 : (_temp_var_256 == 1 ? indices.field_1 : (_temp_var_256 == 2 ? indices.field_2 : (_temp_var_256 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_255 == 0 ? indices.field_0 : (_temp_var_255 == 1 ? indices.field_1 : (_temp_var_255 == 2 ? indices.field_2 : (_temp_var_255 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_254 == 0 ? indices.field_0 : (_temp_var_254 == 1 ? indices.field_1 : (_temp_var_254 == 2 ? indices.field_2 : (_temp_var_254 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_507(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_510)
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
        
        temp_stencil_512 = _block_k_4_(_env_, _kernel_result_510[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_510[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_510[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_510[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_512 = 37;
    }
        
        _result_[_tid_] = temp_stencil_512;
    }
}




__global__ void kernel_515(environment_t *_env_, int _num_threads_, int *_result_, int *_array_517_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_517_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_257 = ((({ int _temp_var_258 = ((({ int _temp_var_259 = ((values[2] % 4));
        (_temp_var_259 == 0 ? indices.field_0 : (_temp_var_259 == 1 ? indices.field_1 : (_temp_var_259 == 2 ? indices.field_2 : (_temp_var_259 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_258 == 0 ? indices.field_0 : (_temp_var_258 == 1 ? indices.field_1 : (_temp_var_258 == 2 ? indices.field_2 : (_temp_var_258 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_257 == 0 ? indices.field_0 : (_temp_var_257 == 1 ? indices.field_1 : (_temp_var_257 == 2 ? indices.field_2 : (_temp_var_257 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_513(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_516)
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
        
        temp_stencil_518 = _block_k_4_(_env_, _kernel_result_516[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_516[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_516[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_516[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_518 = 37;
    }
        
        _result_[_tid_] = temp_stencil_518;
    }
}




__global__ void kernel_521(environment_t *_env_, int _num_threads_, int *_result_, int *_array_523_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_523_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_260 = ((({ int _temp_var_261 = ((({ int _temp_var_262 = ((values[2] % 4));
        (_temp_var_262 == 0 ? indices.field_0 : (_temp_var_262 == 1 ? indices.field_1 : (_temp_var_262 == 2 ? indices.field_2 : (_temp_var_262 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_261 == 0 ? indices.field_0 : (_temp_var_261 == 1 ? indices.field_1 : (_temp_var_261 == 2 ? indices.field_2 : (_temp_var_261 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_260 == 0 ? indices.field_0 : (_temp_var_260 == 1 ? indices.field_1 : (_temp_var_260 == 2 ? indices.field_2 : (_temp_var_260 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_519(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_522)
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
        
        temp_stencil_524 = _block_k_4_(_env_, _kernel_result_522[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_522[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_522[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_522[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_524 = 37;
    }
        
        _result_[_tid_] = temp_stencil_524;
    }
}




__global__ void kernel_527(environment_t *_env_, int _num_threads_, int *_result_, int *_array_529_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_529_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_263 = ((({ int _temp_var_264 = ((({ int _temp_var_265 = ((values[2] % 4));
        (_temp_var_265 == 0 ? indices.field_0 : (_temp_var_265 == 1 ? indices.field_1 : (_temp_var_265 == 2 ? indices.field_2 : (_temp_var_265 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_264 == 0 ? indices.field_0 : (_temp_var_264 == 1 ? indices.field_1 : (_temp_var_264 == 2 ? indices.field_2 : (_temp_var_264 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_263 == 0 ? indices.field_0 : (_temp_var_263 == 1 ? indices.field_1 : (_temp_var_263 == 2 ? indices.field_2 : (_temp_var_263 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_525(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_528)
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
        
        temp_stencil_530 = _block_k_4_(_env_, _kernel_result_528[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_528[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_528[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_528[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_530 = 37;
    }
        
        _result_[_tid_] = temp_stencil_530;
    }
}




__global__ void kernel_533(environment_t *_env_, int _num_threads_, int *_result_, int *_array_535_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_535_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_266 = ((({ int _temp_var_267 = ((({ int _temp_var_268 = ((values[2] % 4));
        (_temp_var_268 == 0 ? indices.field_0 : (_temp_var_268 == 1 ? indices.field_1 : (_temp_var_268 == 2 ? indices.field_2 : (_temp_var_268 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_267 == 0 ? indices.field_0 : (_temp_var_267 == 1 ? indices.field_1 : (_temp_var_267 == 2 ? indices.field_2 : (_temp_var_267 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_266 == 0 ? indices.field_0 : (_temp_var_266 == 1 ? indices.field_1 : (_temp_var_266 == 2 ? indices.field_2 : (_temp_var_266 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_531(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_534)
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
        
        temp_stencil_536 = _block_k_4_(_env_, _kernel_result_534[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_534[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_534[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_534[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_536 = 37;
    }
        
        _result_[_tid_] = temp_stencil_536;
    }
}




__global__ void kernel_539(environment_t *_env_, int _num_threads_, int *_result_, int *_array_541_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_541_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_269 = ((({ int _temp_var_270 = ((({ int _temp_var_271 = ((values[2] % 4));
        (_temp_var_271 == 0 ? indices.field_0 : (_temp_var_271 == 1 ? indices.field_1 : (_temp_var_271 == 2 ? indices.field_2 : (_temp_var_271 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_270 == 0 ? indices.field_0 : (_temp_var_270 == 1 ? indices.field_1 : (_temp_var_270 == 2 ? indices.field_2 : (_temp_var_270 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_269 == 0 ? indices.field_0 : (_temp_var_269 == 1 ? indices.field_1 : (_temp_var_269 == 2 ? indices.field_2 : (_temp_var_269 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_537(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_540)
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
        
        temp_stencil_542 = _block_k_4_(_env_, _kernel_result_540[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_540[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_540[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_540[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_542 = 37;
    }
        
        _result_[_tid_] = temp_stencil_542;
    }
}




__global__ void kernel_545(environment_t *_env_, int _num_threads_, int *_result_, int *_array_547_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_547_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_272 = ((({ int _temp_var_273 = ((({ int _temp_var_274 = ((values[2] % 4));
        (_temp_var_274 == 0 ? indices.field_0 : (_temp_var_274 == 1 ? indices.field_1 : (_temp_var_274 == 2 ? indices.field_2 : (_temp_var_274 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_273 == 0 ? indices.field_0 : (_temp_var_273 == 1 ? indices.field_1 : (_temp_var_273 == 2 ? indices.field_2 : (_temp_var_273 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_272 == 0 ? indices.field_0 : (_temp_var_272 == 1 ? indices.field_1 : (_temp_var_272 == 2 ? indices.field_2 : (_temp_var_272 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_543(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_546)
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
        
        temp_stencil_548 = _block_k_4_(_env_, _kernel_result_546[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_546[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_546[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_546[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_548 = 37;
    }
        
        _result_[_tid_] = temp_stencil_548;
    }
}




__global__ void kernel_551(environment_t *_env_, int _num_threads_, int *_result_, int *_array_553_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_553_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_275 = ((({ int _temp_var_276 = ((({ int _temp_var_277 = ((values[2] % 4));
        (_temp_var_277 == 0 ? indices.field_0 : (_temp_var_277 == 1 ? indices.field_1 : (_temp_var_277 == 2 ? indices.field_2 : (_temp_var_277 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_276 == 0 ? indices.field_0 : (_temp_var_276 == 1 ? indices.field_1 : (_temp_var_276 == 2 ? indices.field_2 : (_temp_var_276 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_275 == 0 ? indices.field_0 : (_temp_var_275 == 1 ? indices.field_1 : (_temp_var_275 == 2 ? indices.field_2 : (_temp_var_275 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_549(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_552)
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
        
        temp_stencil_554 = _block_k_4_(_env_, _kernel_result_552[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_552[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_552[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_552[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_554 = 37;
    }
        
        _result_[_tid_] = temp_stencil_554;
    }
}




__global__ void kernel_557(environment_t *_env_, int _num_threads_, int *_result_, int *_array_559_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_559_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_278 = ((({ int _temp_var_279 = ((({ int _temp_var_280 = ((values[2] % 4));
        (_temp_var_280 == 0 ? indices.field_0 : (_temp_var_280 == 1 ? indices.field_1 : (_temp_var_280 == 2 ? indices.field_2 : (_temp_var_280 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_279 == 0 ? indices.field_0 : (_temp_var_279 == 1 ? indices.field_1 : (_temp_var_279 == 2 ? indices.field_2 : (_temp_var_279 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_278 == 0 ? indices.field_0 : (_temp_var_278 == 1 ? indices.field_1 : (_temp_var_278 == 2 ? indices.field_2 : (_temp_var_278 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_555(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_558)
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
        
        temp_stencil_560 = _block_k_4_(_env_, _kernel_result_558[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_558[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_558[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_558[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_560 = 37;
    }
        
        _result_[_tid_] = temp_stencil_560;
    }
}




__global__ void kernel_563(environment_t *_env_, int _num_threads_, int *_result_, int *_array_565_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_565_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_281 = ((({ int _temp_var_282 = ((({ int _temp_var_283 = ((values[2] % 4));
        (_temp_var_283 == 0 ? indices.field_0 : (_temp_var_283 == 1 ? indices.field_1 : (_temp_var_283 == 2 ? indices.field_2 : (_temp_var_283 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_282 == 0 ? indices.field_0 : (_temp_var_282 == 1 ? indices.field_1 : (_temp_var_282 == 2 ? indices.field_2 : (_temp_var_282 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_281 == 0 ? indices.field_0 : (_temp_var_281 == 1 ? indices.field_1 : (_temp_var_281 == 2 ? indices.field_2 : (_temp_var_281 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_561(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_564)
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
        
        temp_stencil_566 = _block_k_4_(_env_, _kernel_result_564[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_564[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_564[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_564[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_566 = 37;
    }
        
        _result_[_tid_] = temp_stencil_566;
    }
}




__global__ void kernel_569(environment_t *_env_, int _num_threads_, int *_result_, int *_array_571_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_571_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_284 = ((({ int _temp_var_285 = ((({ int _temp_var_286 = ((values[2] % 4));
        (_temp_var_286 == 0 ? indices.field_0 : (_temp_var_286 == 1 ? indices.field_1 : (_temp_var_286 == 2 ? indices.field_2 : (_temp_var_286 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_285 == 0 ? indices.field_0 : (_temp_var_285 == 1 ? indices.field_1 : (_temp_var_285 == 2 ? indices.field_2 : (_temp_var_285 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_284 == 0 ? indices.field_0 : (_temp_var_284 == 1 ? indices.field_1 : (_temp_var_284 == 2 ? indices.field_2 : (_temp_var_284 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_567(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_570)
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
        
        temp_stencil_572 = _block_k_4_(_env_, _kernel_result_570[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_570[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_570[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_570[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_572 = 37;
    }
        
        _result_[_tid_] = temp_stencil_572;
    }
}




__global__ void kernel_575(environment_t *_env_, int _num_threads_, int *_result_, int *_array_577_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_577_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_287 = ((({ int _temp_var_288 = ((({ int _temp_var_289 = ((values[2] % 4));
        (_temp_var_289 == 0 ? indices.field_0 : (_temp_var_289 == 1 ? indices.field_1 : (_temp_var_289 == 2 ? indices.field_2 : (_temp_var_289 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_288 == 0 ? indices.field_0 : (_temp_var_288 == 1 ? indices.field_1 : (_temp_var_288 == 2 ? indices.field_2 : (_temp_var_288 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_287 == 0 ? indices.field_0 : (_temp_var_287 == 1 ? indices.field_1 : (_temp_var_287 == 2 ? indices.field_2 : (_temp_var_287 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_573(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_576)
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
        
        temp_stencil_578 = _block_k_4_(_env_, _kernel_result_576[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_576[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_576[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_576[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_578 = 37;
    }
        
        _result_[_tid_] = temp_stencil_578;
    }
}




__global__ void kernel_581(environment_t *_env_, int _num_threads_, int *_result_, int *_array_583_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_583_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_290 = ((({ int _temp_var_291 = ((({ int _temp_var_292 = ((values[2] % 4));
        (_temp_var_292 == 0 ? indices.field_0 : (_temp_var_292 == 1 ? indices.field_1 : (_temp_var_292 == 2 ? indices.field_2 : (_temp_var_292 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_291 == 0 ? indices.field_0 : (_temp_var_291 == 1 ? indices.field_1 : (_temp_var_291 == 2 ? indices.field_2 : (_temp_var_291 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_290 == 0 ? indices.field_0 : (_temp_var_290 == 1 ? indices.field_1 : (_temp_var_290 == 2 ? indices.field_2 : (_temp_var_290 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_579(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_582)
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
        
        temp_stencil_584 = _block_k_4_(_env_, _kernel_result_582[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_582[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_582[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_582[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_584 = 37;
    }
        
        _result_[_tid_] = temp_stencil_584;
    }
}




__global__ void kernel_587(environment_t *_env_, int _num_threads_, int *_result_, int *_array_589_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_589_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_293 = ((({ int _temp_var_294 = ((({ int _temp_var_295 = ((values[2] % 4));
        (_temp_var_295 == 0 ? indices.field_0 : (_temp_var_295 == 1 ? indices.field_1 : (_temp_var_295 == 2 ? indices.field_2 : (_temp_var_295 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_294 == 0 ? indices.field_0 : (_temp_var_294 == 1 ? indices.field_1 : (_temp_var_294 == 2 ? indices.field_2 : (_temp_var_294 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_293 == 0 ? indices.field_0 : (_temp_var_293 == 1 ? indices.field_1 : (_temp_var_293 == 2 ? indices.field_2 : (_temp_var_293 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_585(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_588)
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
        
        temp_stencil_590 = _block_k_4_(_env_, _kernel_result_588[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_588[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_588[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_588[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_590 = 37;
    }
        
        _result_[_tid_] = temp_stencil_590;
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
    array_command_4 * _ssa_var_base_11;
    array_command_3 * _ssa_var_base_10;
    array_command_3 * _ssa_var_base_9;
    array_command_3 * _ssa_var_base_8;
    array_command_3 * _ssa_var_base_7;
    array_command_3 * _ssa_var_base_6;
    array_command_3 * _ssa_var_base_5;
    array_command_3 * _ssa_var_base_4;
    array_command_3 * _ssa_var_base_3;
    array_command_3 * _ssa_var_base_2;
    array_command_3 * _ssa_var_base_1;
    {
        _ssa_var_base_1 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
        
            array_command_2 * cmd = base;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_474;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_474, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_474);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_473<<<39063, 256>>>(dev_env, 10000000, _kernel_result_474);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_474;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_2 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_1);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_480;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_480, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_480);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_479<<<39063, 256>>>(dev_env, 10000000, _kernel_result_480, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_478;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_478, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_478);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_477<<<39063, 256>>>(dev_env, 10000000, _kernel_result_478, _kernel_result_480);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_478;
        
                    timeStartMeasure();
        
            if (_kernel_result_480 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_480));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_480),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_3 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_2);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_492;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_492, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_492);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_491<<<39063, 256>>>(dev_env, 10000000, _kernel_result_492, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_490;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_490, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_490);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_489<<<39063, 256>>>(dev_env, 10000000, _kernel_result_490, _kernel_result_492);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_490;
        
                    timeStartMeasure();
        
            if (_kernel_result_492 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_492));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_492),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_4 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_3);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_504;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_504, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_504);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_503<<<39063, 256>>>(dev_env, 10000000, _kernel_result_504, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_502;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_502, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_502);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_501<<<39063, 256>>>(dev_env, 10000000, _kernel_result_502, _kernel_result_504);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_502;
        
                    timeStartMeasure();
        
            if (_kernel_result_504 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_504));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_504),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_5 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_4);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_516;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_516, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_516);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_515<<<39063, 256>>>(dev_env, 10000000, _kernel_result_516, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_514;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_514, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_514);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_513<<<39063, 256>>>(dev_env, 10000000, _kernel_result_514, _kernel_result_516);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_514;
        
                    timeStartMeasure();
        
            if (_kernel_result_516 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_516));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_516),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_6 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_5);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_528;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_528, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_528);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_527<<<39063, 256>>>(dev_env, 10000000, _kernel_result_528, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_526;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_526, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_526);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_525<<<39063, 256>>>(dev_env, 10000000, _kernel_result_526, _kernel_result_528);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_526;
        
                    timeStartMeasure();
        
            if (_kernel_result_528 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_528));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_528),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_7 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_6);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_540;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_540, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_540);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_539<<<39063, 256>>>(dev_env, 10000000, _kernel_result_540, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_538;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_538, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_538);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_537<<<39063, 256>>>(dev_env, 10000000, _kernel_result_538, _kernel_result_540);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_538;
        
                    timeStartMeasure();
        
            if (_kernel_result_540 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_540));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_540),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_8 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_7);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_552;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_552, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_552);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_551<<<39063, 256>>>(dev_env, 10000000, _kernel_result_552, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_550;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_550, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_550);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_549<<<39063, 256>>>(dev_env, 10000000, _kernel_result_550, _kernel_result_552);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_550;
        
                    timeStartMeasure();
        
            if (_kernel_result_552 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_552));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_552),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_9 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_8);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_564;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_564, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_564);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_563<<<39063, 256>>>(dev_env, 10000000, _kernel_result_564, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_562;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_562, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_562);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_561<<<39063, 256>>>(dev_env, 10000000, _kernel_result_562, _kernel_result_564);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_562;
        
                    timeStartMeasure();
        
            if (_kernel_result_564 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_564));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_564),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_10 = new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = new array_command_4(NULL, _ssa_var_base_9);
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_576;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_576, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_576);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_575<<<39063, 256>>>(dev_env, 10000000, _kernel_result_576, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_574;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_574, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_574);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_573<<<39063, 256>>>(dev_env, 10000000, _kernel_result_574, _kernel_result_576);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_574;
        
                    timeStartMeasure();
        
            if (_kernel_result_576 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_576));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_576),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }));
        _ssa_var_base_11 = new array_command_4(NULL, _ssa_var_base_10);
        return ({
            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_base_1].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
        
            array_command_4 * cmd = _ssa_var_base_11;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_588;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_588, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_588);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_587<<<39063, 256>>>(dev_env, 10000000, _kernel_result_588, ((int *) ((int *) cmd->input_0->input_0.content)));
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);    timeStartMeasure();
            int * _kernel_result_586;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_586, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_586);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_585<<<39063, 256>>>(dev_env, 10000000, _kernel_result_586, _kernel_result_588);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_586;
        
                    timeStartMeasure();
        
            if (_kernel_result_588 != cmd->result) {
                // Don't free memory if it is the result. There is already a similar check in
                // program_builder (free all except for last). However, this check is not sufficient in
                // case the same array is reused!
        
                checkErrorReturn(program_result, cudaFree(_kernel_result_588));
                // Remove from list of allocations
                program_result->device_allocations->erase(
                    std::remove(
                        program_result->device_allocations->begin(),
                        program_result->device_allocations->end(),
                        _kernel_result_588),
                    program_result->device_allocations->end());
            }
        
            timeReportMeasure(program_result, free_memory);
        
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        });
    }
}

#undef checkErrorReturn
#define checkErrorReturn(result_var, expr) \
expr

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
