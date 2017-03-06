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
struct array_command_8 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_8(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_14 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_14(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_27 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_14 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_27(int *result = NULL, array_command_14 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_4(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_6(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_11 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_11(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_19 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_14 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_19(int *result = NULL, array_command_14 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_23 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_14 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_23(int *result = NULL, array_command_14 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
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
        return (indices.field_2 % 133777);
    }
}

#endif


__global__ void kernel_377(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (indices.field_2 % 133777);
    }
}

#endif


__global__ void kernel_379(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_381(environment_t *_env_, int _num_threads_, int *_result_, int *_array_383_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_383_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_384(environment_t *_env_, int _num_threads_, int *_result_, int *_array_386_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_27_(_env_, _block_k_14_(_env_, _array_386_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_387(environment_t *_env_, int _num_threads_, int *_result_, int *_array_389_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_389_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_390(environment_t *_env_, int _num_threads_, int *_result_, int *_array_392_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_27_(_env_, _block_k_14_(_env_, _array_392_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_393(environment_t *_env_, int _num_threads_, int *_result_, int *_array_395_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_395_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_396(environment_t *_env_, int _num_threads_, int *_result_, int *_array_398_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_27_(_env_, _block_k_14_(_env_, _array_398_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_399(environment_t *_env_, int _num_threads_, int *_result_, int *_array_401_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_401_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_402(environment_t *_env_, int _num_threads_, int *_result_, int *_array_404_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_27_(_env_, _block_k_14_(_env_, _array_404_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_405(environment_t *_env_, int _num_threads_, int *_result_, int *_array_407_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_4_(_env_, _array_407_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_408(environment_t *_env_, int _num_threads_, int *_result_, int *_array_410_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_410_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_411(environment_t *_env_, int _num_threads_, int *_result_, int *_array_413_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_413_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_414(environment_t *_env_, int _num_threads_, int *_result_, int *_array_416_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_19_(_env_, _block_k_14_(_env_, _array_416_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_417(environment_t *_env_, int _num_threads_, int *_result_, int *_array_419_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_23_(_env_, _block_k_14_(_env_, _array_419_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_420(environment_t *_env_, int _num_threads_, int *_result_, int *_array_422_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_4_(_env_, _array_422_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_423(environment_t *_env_, int _num_threads_, int *_result_, int *_array_425_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_425_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_426(environment_t *_env_, int _num_threads_, int *_result_, int *_array_428_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_428_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_429(environment_t *_env_, int _num_threads_, int *_result_, int *_array_431_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_19_(_env_, _block_k_14_(_env_, _array_431_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_432(environment_t *_env_, int _num_threads_, int *_result_, int *_array_434_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_23_(_env_, _block_k_14_(_env_, _array_434_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_435(environment_t *_env_, int _num_threads_, int *_result_, int *_array_437_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_4_(_env_, _array_437_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_438(environment_t *_env_, int _num_threads_, int *_result_, int *_array_440_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_440_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_441(environment_t *_env_, int _num_threads_, int *_result_, int *_array_443_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_443_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_444(environment_t *_env_, int _num_threads_, int *_result_, int *_array_446_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_19_(_env_, _block_k_14_(_env_, _array_446_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_447(environment_t *_env_, int _num_threads_, int *_result_, int *_array_449_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_23_(_env_, _block_k_14_(_env_, _array_449_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ int _block_k_4_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_450(environment_t *_env_, int _num_threads_, int *_result_, int *_array_452_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_4_(_env_, _array_452_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_453(environment_t *_env_, int _num_threads_, int *_result_, int *_array_455_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_455_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_456(environment_t *_env_, int _num_threads_, int *_result_, int *_array_458_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_458_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_459(environment_t *_env_, int _num_threads_, int *_result_, int *_array_461_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_19_(_env_, _block_k_14_(_env_, _array_461_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_462(environment_t *_env_, int _num_threads_, int *_result_, int *_array_464_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_23_(_env_, _block_k_14_(_env_, _array_464_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_465(environment_t *_env_, int _num_threads_, int *_result_, int *_array_467_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_467_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_468(environment_t *_env_, int _num_threads_, int *_result_, int *_array_470_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _array_470_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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
    array_command_14 * _ssa_var_old_data_16;
    array_command_14 * _ssa_var_y_15;
    array_command_11 * _ssa_var_old_data_12;
    array_command_11 * _ssa_var_y_11;
    union_t _ssa_var_old_data_10;
    union_t _ssa_var_y_9;
    union_t _ssa_var_old_data_14;
    union_t _ssa_var_y_13;
    union_t _ssa_var_old_data_6;
    union_t _ssa_var_y_5;
    union_t _ssa_var_old_data_8;
    union_t _ssa_var_y_7;
    union_t _ssa_var_old_data_4;
    union_t _ssa_var_y_3;
    int r;
    union_t _ssa_var_old_data_2;
    union_t _ssa_var_y_1;
    {
        _ssa_var_y_1 = union_t(10, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
        
            array_command_2 * cmd = x;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_378;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_378, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_378);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_377<<<39063, 256>>>(dev_env, 10000000, _kernel_result_378);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_378;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }))));
        _ssa_var_old_data_2 = union_t(47, union_v_t::from_pointer((void *) x));
        for (r = 0; r <= (200 - 1); r++)
        {
            if (((((r % 2)) == 0)))
            {
                if (((((r % 3)) == 0)))
                {
                    _ssa_var_y_3 = ({
                        union_t _polytemp_result_137;
                        {
                            union_t _polytemp_expr_138 = _ssa_var_y_1;
                            switch (_polytemp_expr_138.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = union_t(48, union_v_t::from_pointer((void *) new array_command_4(NULL, (array_command_3 *) _polytemp_expr_138.value.pointer))); break;
                                case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = union_t(50, union_v_t::from_pointer((void *) new array_command_19(NULL, (array_command_14 *) _polytemp_expr_138.value.pointer))); break;
                            }
                        }
                        _polytemp_result_137;
                    });
                    ({
                        bool _polytemp_result_139;
                        {
                            union_t _polytemp_expr_140 = _ssa_var_old_data_2;
                            switch (_polytemp_expr_140.class_id)
                            {
                                case 47: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_139 = ({
                                    array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_140.value.pointer;
                                
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
                                case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_139 = ({
                                    array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_140.value.pointer;
                                
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
                        _polytemp_result_139;
                    });
                    _ssa_var_old_data_4 = _ssa_var_y_3;
                    _ssa_var_y_7 = _ssa_var_y_3;
                    _ssa_var_old_data_8 = _ssa_var_old_data_4;
                }
                else
                {
                    _ssa_var_y_5 = ({
                        union_t _polytemp_result_141;
                        {
                            union_t _polytemp_expr_142 = _ssa_var_y_1;
                            switch (_polytemp_expr_142.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_141 = union_t(51, union_v_t::from_pointer((void *) new array_command_6(NULL, (array_command_3 *) _polytemp_expr_142.value.pointer))); break;
                                case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_141 = union_t(52, union_v_t::from_pointer((void *) new array_command_23(NULL, (array_command_14 *) _polytemp_expr_142.value.pointer))); break;
                            }
                        }
                        _polytemp_result_141;
                    });
                    ({
                        bool _polytemp_result_143;
                        {
                            union_t _polytemp_expr_144 = _ssa_var_old_data_2;
                            switch (_polytemp_expr_144.class_id)
                            {
                                case 47: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_143 = ({
                                    array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_144.value.pointer;
                                
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
                                case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_143 = ({
                                    array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_144.value.pointer;
                                
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
                        _polytemp_result_143;
                    });
                    _ssa_var_old_data_6 = _ssa_var_y_5;
                    _ssa_var_y_7 = _ssa_var_y_5;
                    _ssa_var_old_data_8 = _ssa_var_old_data_6;
                }
                _ssa_var_y_13 = _ssa_var_y_7;
                _ssa_var_old_data_14 = _ssa_var_old_data_8;
            }
            else
            {
                _ssa_var_y_9 = ({
                    union_t _polytemp_result_145;
                    {
                        union_t _polytemp_expr_146 = _ssa_var_y_1;
                        switch (_polytemp_expr_146.class_id)
                        {
                            case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = union_t(53, union_v_t::from_pointer((void *) new array_command_8(NULL, (array_command_3 *) _polytemp_expr_146.value.pointer))); break;
                            case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = union_t(54, union_v_t::from_pointer((void *) new array_command_27(NULL, (array_command_14 *) _polytemp_expr_146.value.pointer))); break;
                        }
                    }
                    _polytemp_result_145;
                });
                ({
                    bool _polytemp_result_147;
                    {
                        union_t _polytemp_expr_148 = _ssa_var_old_data_2;
                        switch (_polytemp_expr_148.class_id)
                        {
                            case 47: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_147 = ({
                                array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_148.value.pointer;
                            
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
                            case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_147 = ({
                                array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_148.value.pointer;
                            
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
                    _polytemp_result_147;
                });
                _ssa_var_old_data_10 = _ssa_var_y_9;
                _ssa_var_y_11 = new array_command_11(NULL, new array_command_3(NULL, ({
                    variable_size_array_t _polytemp_result_149;
                    {
                        union_t _polytemp_expr_150 = _ssa_var_y_9;
                        switch (_polytemp_expr_150.class_id)
                        {
                            case 53: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_149 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_8 * cmd = (array_command_8 *) _polytemp_expr_150.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_382;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_382, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_382);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_381<<<39063, 256>>>(dev_env, 10000000, _kernel_result_382, ((int *) ((int *) cmd->input_0->input_0.content)));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_382;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 54: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_149 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_27 * cmd = (array_command_27 *) _polytemp_expr_150.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_385;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_385, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_385);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_384<<<39063, 256>>>(dev_env, 10000000, _kernel_result_385, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_385;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                        }
                    }
                    _polytemp_result_149;
                })));
                ({
                    bool _polytemp_result_157;
                    {
                        union_t _polytemp_expr_158 = _ssa_var_old_data_10;
                        switch (_polytemp_expr_158.class_id)
                        {
                            case 53: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                                array_command_8 * cmd_to_free = (array_command_8 *) _polytemp_expr_158.value.pointer;
                            
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
                            case 54: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                                array_command_27 * cmd_to_free = (array_command_27 *) _polytemp_expr_158.value.pointer;
                            
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
                    _polytemp_result_157;
                });
                _ssa_var_old_data_12 = _ssa_var_y_11;
                _ssa_var_y_13 = union_t(55, union_v_t::from_pointer((void *) _ssa_var_y_11));
                _ssa_var_old_data_14 = union_t(55, union_v_t::from_pointer((void *) _ssa_var_old_data_12));
            }
            _ssa_var_y_15 = new array_command_14(NULL, new array_command_3(NULL, ({
                variable_size_array_t _polytemp_result_159;
                {
                    union_t _polytemp_expr_160 = _ssa_var_y_13;
                    switch (_polytemp_expr_160.class_id)
                    {
                        case 48: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_160.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_406;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_406, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_406);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_405<<<39063, 256>>>(dev_env, 10000000, _kernel_result_406, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_406;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 51: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_6 * cmd = (array_command_6 *) _polytemp_expr_160.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_409;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_409, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_409);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_408<<<39063, 256>>>(dev_env, 10000000, _kernel_result_409, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_409;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 55: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_9].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_11 * cmd = (array_command_11 *) _polytemp_expr_160.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_412;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_412, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_412);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_411<<<39063, 256>>>(dev_env, 10000000, _kernel_result_412, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_412;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 50: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_19 * cmd = (array_command_19 *) _polytemp_expr_160.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_415;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_415, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_415);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_414<<<39063, 256>>>(dev_env, 10000000, _kernel_result_415, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_415;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 52: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_1].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_23 * cmd = (array_command_23 *) _polytemp_expr_160.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_418;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_418, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_418);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_417<<<39063, 256>>>(dev_env, 10000000, _kernel_result_418, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_418;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                    }
                }
                _polytemp_result_159;
            })));
            ({
                bool _polytemp_result_167;
                {
                    union_t _polytemp_expr_168 = _ssa_var_old_data_14;
                    switch (_polytemp_expr_168.class_id)
                    {
                        case 48: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_167 = ({
                            array_command_4 * cmd_to_free = (array_command_4 *) _polytemp_expr_168.value.pointer;
                        
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
                        case 51: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_167 = ({
                            array_command_6 * cmd_to_free = (array_command_6 *) _polytemp_expr_168.value.pointer;
                        
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
                        case 55: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_167 = ({
                            array_command_11 * cmd_to_free = (array_command_11 *) _polytemp_expr_168.value.pointer;
                        
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
                        case 50: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_167 = ({
                            array_command_19 * cmd_to_free = (array_command_19 *) _polytemp_expr_168.value.pointer;
                        
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
                        case 52: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_167 = ({
                            array_command_23 * cmd_to_free = (array_command_23 *) _polytemp_expr_168.value.pointer;
                        
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
                _polytemp_result_167;
            });
            _ssa_var_old_data_16 = _ssa_var_y_15;
            _ssa_var_y_1 = union_t(49, union_v_t::from_pointer((void *) _ssa_var_y_15));
            _ssa_var_old_data_2 = union_t(49, union_v_t::from_pointer((void *) _ssa_var_old_data_16));
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_169;
            {
                union_t _polytemp_expr_170 = _ssa_var_y_1;
                switch (_polytemp_expr_170.class_id)
                {
                    case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_169 = ({
                        // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_170.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_466;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_466, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_466);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_465<<<39063, 256>>>(dev_env, 10000000, _kernel_result_466, ((int *) cmd->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_466;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 49: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_169 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_13].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_14 * cmd = (array_command_14 *) _polytemp_expr_170.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_469;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_469, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_469);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_468<<<39063, 256>>>(dev_env, 10000000, _kernel_result_469, ((int *) ((int *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_469;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                }
            }
            _polytemp_result_169;
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
