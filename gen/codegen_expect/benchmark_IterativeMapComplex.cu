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
struct array_command_5 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_4_lt_int_int_int_int_gt_t *result;
    __host__ __device__ array_command_5(indexed_struct_4_lt_int_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_2 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_4(int *result = NULL, array_command_2 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_12 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_4 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_12(int *result = NULL, array_command_4 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_2 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_6(int *result = NULL, array_command_2 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_14 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_6 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_14(int *result = NULL, array_command_6 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_8 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_2 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_8(int *result = NULL, array_command_2 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_10 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_8 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_10(int *result = NULL, array_command_8 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_16 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_10 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_16(int *result = NULL, array_command_10 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_3 {
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    int *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_3(int *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_23 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_23(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_37 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_23 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_37(int *result = NULL, array_command_23 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_52 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_52(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_66 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_52 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_66(int *result = NULL, array_command_52 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_89 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_89(int *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_93 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_89 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_93(int *result = NULL, array_command_89 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_105 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_93 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_105(int *result = NULL, array_command_93 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
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


__global__ void kernel_901(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_903(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_905(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_907(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_909(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_911(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_913(environment_t *_env_, int _num_threads_, int *_result_, int *_array_915_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_915_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_916(environment_t *_env_, int _num_threads_, int *_result_, int *_array_918_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_918_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_919(environment_t *_env_, int _num_threads_, int *_result_, int *_array_921_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_921_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_922(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_924(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_926(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_928(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_930(environment_t *_env_, int _num_threads_, int *_result_, int *_array_932_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_932_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_933(environment_t *_env_, int _num_threads_, int *_result_, int *_array_935_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_935_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_936(environment_t *_env_, int _num_threads_, int *_result_, int *_array_938_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_938_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_939(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_941(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_943(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_945(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_947(environment_t *_env_, int _num_threads_, int *_result_, int *_array_949_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_949_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_950(environment_t *_env_, int _num_threads_, int *_result_, int *_array_952_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_952_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_953(environment_t *_env_, int _num_threads_, int *_result_, int *_array_955_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_955_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_956(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_958(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_960(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_962(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_964(environment_t *_env_, int _num_threads_, int *_result_, int *_array_966_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_966_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_967(environment_t *_env_, int _num_threads_, int *_result_, int *_array_969_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_969_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_970(environment_t *_env_, int _num_threads_, int *_result_, int *_array_972_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_972_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_973(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_975(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_977(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_979(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_981(environment_t *_env_, int _num_threads_, int *_result_, int *_array_983_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_983_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_984(environment_t *_env_, int _num_threads_, int *_result_, int *_array_986_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_986_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_987(environment_t *_env_, int _num_threads_, int *_result_, int *_array_989_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_989_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_990(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_992(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_994(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_996(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_998(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1000_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1000_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1001(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1003_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1003_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1004(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1006_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1006_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1007(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1009(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1011(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1013(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1015(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1017_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1017_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1018(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1020_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1020_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1021(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1023_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1023_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1024(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1026(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1028(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1030(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1032(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1034_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1034_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1035(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1037_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1037_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1038(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1040_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1040_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1041(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1043(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1045(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1047(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1049(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1051_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1051_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1052(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1054_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1054_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1055(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1057_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1057_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1058(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1060(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1062(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1064(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1066(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1068_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1068_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1069(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1071_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1071_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1072(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1074_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1074_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1075(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1077(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1079(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1081(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1083(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1085_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1085_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1086(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1088_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1088_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1089(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1091_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1091_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1092(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1094(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1096(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1098(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1100(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1102_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1102_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1103(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1105_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1105_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1106(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1108_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1108_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1109(environment_t *_env_, int _num_threads_, int *_result_)
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



// TODO: There should be a better to check if _block_k_12_ is already defined
#ifndef _block_k_12__func
#define _block_k_12__func
__device__ int _block_k_12_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1111(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_12_(_env_, _block_k_4_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_1113(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _block_k_6_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_10_ is already defined
#ifndef _block_k_10__func
#define _block_k_10__func
__device__ int _block_k_10_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_16_ is already defined
#ifndef _block_k_16__func
#define _block_k_16__func
__device__ int _block_k_16_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1115(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_16_(_env_, _block_k_10_(_env_, _block_k_8_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1117(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1119_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_1119_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}





// TODO: There should be a better to check if _block_k_52_ is already defined
#ifndef _block_k_52__func
#define _block_k_52__func
__device__ int _block_k_52_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_66_ is already defined
#ifndef _block_k_66__func
#define _block_k_66__func
__device__ int _block_k_66_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1120(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1122_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_1122_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}







// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1123(environment_t *_env_, int _num_threads_, int *_result_, int *_array_1125_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_1125_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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
    union_t _ssa_var_old_data_10;
    union_t _ssa_var_y_9;
    union_t _ssa_var_y_7;
    union_t _ssa_var_y_6;
    union_t _ssa_var_y_8;
    union_t _ssa_var_y_4;
    union_t _ssa_var_y_5;
    union_t _ssa_var_y_3;
    int r;
    union_t _ssa_var_old_data_2;
    union_t _ssa_var_y_1;
    {
        _ssa_var_y_1 = union_t(71, union_v_t::from_pointer((void *) x));
        _ssa_var_old_data_2 = union_t(11, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
        
            array_command_2 * cmd = x;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_902;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_902, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_902);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_901<<<39063, 256>>>(dev_env, 10000000, _kernel_result_902);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_902;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }))));
        for (r = 0; r <= (200 - 1); r++)
        {
            if (((((r % 2)) == 0)))
            {
                if (((((r % 3)) == 0)))
                {
                    _ssa_var_y_3 = union_t(78, union_v_t::from_pointer((void *) new array_command_23(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_129;
                        {
                            union_t _polytemp_expr_130 = _ssa_var_y_1;
                            switch (_polytemp_expr_130.class_id)
                            {
                                case 71: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                                
                                    array_command_2 * cmd = (array_command_2 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_906;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_906, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_906);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_905<<<39063, 256>>>(dev_env, 10000000, _kernel_result_906);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_906;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 72: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_12 * cmd = (array_command_12 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_908;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_908, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_908);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_907<<<39063, 256>>>(dev_env, 10000000, _kernel_result_908);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_908;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 73: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_14 * cmd = (array_command_14 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_910;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_910, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_910);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_909<<<39063, 256>>>(dev_env, 10000000, _kernel_result_910);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_910;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 74: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_16 * cmd = (array_command_16 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_912;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_912, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_912);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_911<<<39063, 256>>>(dev_env, 10000000, _kernel_result_912);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_912;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 75: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_37 * cmd = (array_command_37 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_914;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_914, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_914);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_913<<<39063, 256>>>(dev_env, 10000000, _kernel_result_914, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_914;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 76: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_66 * cmd = (array_command_66 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_917;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_917, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_917);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_916<<<39063, 256>>>(dev_env, 10000000, _kernel_result_917, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_917;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 77: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_129 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_105 * cmd = (array_command_105 *) _polytemp_expr_130.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_920;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_920, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_920);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_919<<<39063, 256>>>(dev_env, 10000000, _kernel_result_920, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_920;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_129;
                    })))));
                    _ssa_var_y_5 = _ssa_var_y_3;
                }
                else
                {
                    _ssa_var_y_4 = union_t(79, union_v_t::from_pointer((void *) new array_command_52(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_137;
                        {
                            union_t _polytemp_expr_138 = _ssa_var_y_1;
                            switch (_polytemp_expr_138.class_id)
                            {
                                case 71: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                                
                                    array_command_2 * cmd = (array_command_2 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_974;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_974, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_974);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_973<<<39063, 256>>>(dev_env, 10000000, _kernel_result_974);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_974;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 72: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_12 * cmd = (array_command_12 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_976;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_976, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_976);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_975<<<39063, 256>>>(dev_env, 10000000, _kernel_result_976);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_976;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 73: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_14 * cmd = (array_command_14 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_978;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_978, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_978);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_977<<<39063, 256>>>(dev_env, 10000000, _kernel_result_978);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_978;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 74: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_16 * cmd = (array_command_16 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_980;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_980, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_980);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_979<<<39063, 256>>>(dev_env, 10000000, _kernel_result_980);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_980;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 75: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_37 * cmd = (array_command_37 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_982;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_982, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_982);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_981<<<39063, 256>>>(dev_env, 10000000, _kernel_result_982, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_982;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 76: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_66 * cmd = (array_command_66 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_985;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_985, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_985);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_984<<<39063, 256>>>(dev_env, 10000000, _kernel_result_985, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_985;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 77: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_137 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_105 * cmd = (array_command_105 *) _polytemp_expr_138.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_988;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_988, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_988);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_987<<<39063, 256>>>(dev_env, 10000000, _kernel_result_988, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_988;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_137;
                    })))));
                    _ssa_var_y_5 = _ssa_var_y_4;
                }
                _ssa_var_y_8 = _ssa_var_y_5;
            }
            else
            {
                _ssa_var_y_6 = union_t(80, union_v_t::from_pointer((void *) new array_command_89(NULL, new array_command_3(NULL, ({
                    variable_size_array_t _polytemp_result_145;
                    {
                        union_t _polytemp_expr_146 = _ssa_var_y_1;
                        switch (_polytemp_expr_146.class_id)
                        {
                            case 71: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                            
                                array_command_2 * cmd = (array_command_2 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1042;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1042, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1042);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1041<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1042);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1042;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 72: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_12 * cmd = (array_command_12 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1044;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1044, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1044);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1043<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1044);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1044;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 73: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_14 * cmd = (array_command_14 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1046;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1046, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1046);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1045<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1046);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1046;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 74: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_16 * cmd = (array_command_16 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1048;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1048, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1048);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1047<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1048);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1048;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 75: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_37 * cmd = (array_command_37 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1050;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1050, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1050);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1049<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1050, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1050;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 76: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_66 * cmd = (array_command_66 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1053;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1053, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1053);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1052<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1053, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1053;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 77: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_145 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_105 * cmd = (array_command_105 *) _polytemp_expr_146.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_1056;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1056, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_1056);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_1055<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1056, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_1056;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                        }
                    }
                    _polytemp_result_145;
                })))));
                _ssa_var_y_7 = ({
                    union_t _polytemp_result_153;
                    {
                        union_t _polytemp_expr_154 = _ssa_var_y_6;
                        switch (_polytemp_expr_154.class_id)
                        {
                            case 81: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_153 = union_t(82, union_v_t::from_pointer((void *) new array_command_10(NULL, (array_command_8 *) _polytemp_expr_154.value.pointer))); break;
                            case 80: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_153 = union_t(83, union_v_t::from_pointer((void *) new array_command_93(NULL, (array_command_89 *) _polytemp_expr_154.value.pointer))); break;
                        }
                    }
                    _polytemp_result_153;
                });
                _ssa_var_y_8 = _ssa_var_y_7;
            }
            _ssa_var_y_9 = ({
                union_t _polytemp_result_155;
                {
                    union_t _polytemp_expr_156 = _ssa_var_y_8;
                    switch (_polytemp_expr_156.class_id)
                    {
                        case 84: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(72, union_v_t::from_pointer((void *) new array_command_12(NULL, (array_command_4 *) _polytemp_expr_156.value.pointer))); break;
                        case 85: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(73, union_v_t::from_pointer((void *) new array_command_14(NULL, (array_command_6 *) _polytemp_expr_156.value.pointer))); break;
                        case 82: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(74, union_v_t::from_pointer((void *) new array_command_16(NULL, (array_command_10 *) _polytemp_expr_156.value.pointer))); break;
                        case 78: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(75, union_v_t::from_pointer((void *) new array_command_37(NULL, (array_command_23 *) _polytemp_expr_156.value.pointer))); break;
                        case 79: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(76, union_v_t::from_pointer((void *) new array_command_66(NULL, (array_command_52 *) _polytemp_expr_156.value.pointer))); break;
                        case 83: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_155 = union_t(77, union_v_t::from_pointer((void *) new array_command_105(NULL, (array_command_93 *) _polytemp_expr_156.value.pointer))); break;
                    }
                }
                _polytemp_result_155;
            });
            ({
                bool _polytemp_result_157;
                {
                    union_t _polytemp_expr_158 = _ssa_var_old_data_2;
                    switch (_polytemp_expr_158.class_id)
                    {
                        case 11: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_3 * cmd_to_free = (array_command_3 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 72: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_12 * cmd_to_free = (array_command_12 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 73: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 74: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_16 * cmd_to_free = (array_command_16 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 75: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_37 * cmd_to_free = (array_command_37 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 76: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_66 * cmd_to_free = (array_command_66 *) _polytemp_expr_158.value.pointer;
                        
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
                        case 77: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_157 = ({
                            array_command_105 * cmd_to_free = (array_command_105 *) _polytemp_expr_158.value.pointer;
                        
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
            _ssa_var_old_data_10 = _ssa_var_y_9;
            _ssa_var_y_1 = _ssa_var_y_9;
            _ssa_var_old_data_2 = _ssa_var_old_data_10;
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_159;
            {
                union_t _polytemp_expr_160 = _ssa_var_y_1;
                switch (_polytemp_expr_160.class_id)
                {
                    case 71: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1110;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1110, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1110);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1109<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1110);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1110;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 72: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_12 * cmd = (array_command_12 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1112;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1112, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1112);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1111<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1112);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1112;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 73: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_14 * cmd = (array_command_14 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1114;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1114, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1114);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1113<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1114);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1114;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 74: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_16 * cmd = (array_command_16 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1116;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1116, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1116);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1115<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1116);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1116;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 75: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_37 * cmd = (array_command_37 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1118;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1118, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1118);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1117<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1118, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1118;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 76: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_66 * cmd = (array_command_66 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1121;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1121, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1121);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1120<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1121, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1121;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 77: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_159 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_105 * cmd = (array_command_105 *) _polytemp_expr_160.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_1124;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_1124, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_1124);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_1123<<<39063, 256>>>(dev_env, 10000000, _kernel_result_1124, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_1124;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                }
            }
            _polytemp_result_159;
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
