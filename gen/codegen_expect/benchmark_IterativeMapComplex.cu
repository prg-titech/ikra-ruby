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


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_3(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_7(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_9(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_11(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_13(environment_t *_env_, int _num_threads_, int *_result_, int *_array_15_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_15_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_16(environment_t *_env_, int _num_threads_, int *_result_, int *_array_18_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_18_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_19(environment_t *_env_, int _num_threads_, int *_result_, int *_array_21_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_21_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_22(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_24(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_26(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_28(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_30(environment_t *_env_, int _num_threads_, int *_result_, int *_array_32_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_32_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_33(environment_t *_env_, int _num_threads_, int *_result_, int *_array_35_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_35_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_36(environment_t *_env_, int _num_threads_, int *_result_, int *_array_38_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_38_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_39(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_41(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_43(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_45(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_47(environment_t *_env_, int _num_threads_, int *_result_, int *_array_49_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_49_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_50(environment_t *_env_, int _num_threads_, int *_result_, int *_array_52_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_52_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_53(environment_t *_env_, int _num_threads_, int *_result_, int *_array_55_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_55_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_56(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_58(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_60(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_62(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_64(environment_t *_env_, int _num_threads_, int *_result_, int *_array_66_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_66_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_67(environment_t *_env_, int _num_threads_, int *_result_, int *_array_69_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_69_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_70(environment_t *_env_, int _num_threads_, int *_result_, int *_array_72_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_72_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_73(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_75(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_77(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_79(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_81(environment_t *_env_, int _num_threads_, int *_result_, int *_array_83_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_83_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_84(environment_t *_env_, int _num_threads_, int *_result_, int *_array_86_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_86_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_87(environment_t *_env_, int _num_threads_, int *_result_, int *_array_89_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_89_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_90(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_92(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_94(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_96(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_98(environment_t *_env_, int _num_threads_, int *_result_, int *_array_100_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_100_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_101(environment_t *_env_, int _num_threads_, int *_result_, int *_array_103_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_103_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_104(environment_t *_env_, int _num_threads_, int *_result_, int *_array_106_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_106_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_107(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_109(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_111(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_113(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_115(environment_t *_env_, int _num_threads_, int *_result_, int *_array_117_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_117_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_118(environment_t *_env_, int _num_threads_, int *_result_, int *_array_120_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_120_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_121(environment_t *_env_, int _num_threads_, int *_result_, int *_array_123_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_123_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_124(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_126(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_128(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_130(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_132(environment_t *_env_, int _num_threads_, int *_result_, int *_array_134_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_134_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_135(environment_t *_env_, int _num_threads_, int *_result_, int *_array_137_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_137_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_138(environment_t *_env_, int _num_threads_, int *_result_, int *_array_140_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_140_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_141(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_143(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_145(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_147(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_149(environment_t *_env_, int _num_threads_, int *_result_, int *_array_151_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_151_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_152(environment_t *_env_, int _num_threads_, int *_result_, int *_array_154_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_154_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_155(environment_t *_env_, int _num_threads_, int *_result_, int *_array_157_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_157_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_158(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_160(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_162(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_164(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_166(environment_t *_env_, int _num_threads_, int *_result_, int *_array_168_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_168_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_169(environment_t *_env_, int _num_threads_, int *_result_, int *_array_171_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_171_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_172(environment_t *_env_, int _num_threads_, int *_result_, int *_array_174_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_174_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_175(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_177(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_179(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_181(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_183(environment_t *_env_, int _num_threads_, int *_result_, int *_array_185_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_185_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_186(environment_t *_env_, int _num_threads_, int *_result_, int *_array_188_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_188_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_189(environment_t *_env_, int _num_threads_, int *_result_, int *_array_191_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_191_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_192(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_194(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_196(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_198(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_200(environment_t *_env_, int _num_threads_, int *_result_, int *_array_202_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_202_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_203(environment_t *_env_, int _num_threads_, int *_result_, int *_array_205_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_205_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_206(environment_t *_env_, int _num_threads_, int *_result_, int *_array_208_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_208_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_209(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_211(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_213(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_215(environment_t *_env_, int _num_threads_, int *_result_)
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


__global__ void kernel_217(environment_t *_env_, int _num_threads_, int *_result_, int *_array_219_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_37_(_env_, _block_k_23_(_env_, _array_219_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_220(environment_t *_env_, int _num_threads_, int *_result_, int *_array_222_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_66_(_env_, _block_k_52_(_env_, _array_222_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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


__global__ void kernel_223(environment_t *_env_, int _num_threads_, int *_result_, int *_array_225_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_105_(_env_, _block_k_93_(_env_, _block_k_89_(_env_, _array_225_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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
        _ssa_var_y_1 = union_t(10, union_v_t::from_pointer((void *) x));
        _ssa_var_old_data_2 = union_t(11, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
        
            array_command_2 * cmd = x;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_2;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_2);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_1<<<39063, 256>>>(dev_env, 10000000, _kernel_result_2);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_2;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }))));
        for (r = 0; r <= (200 - 1); r++)
        {
            if (((((r % 2)) == 0)))
            {
                if (((((r % 3)) == 0)))
                {
                    _ssa_var_y_3 = union_t(18, union_v_t::from_pointer((void *) new array_command_23(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_1;
                        {
                            union_t _polytemp_expr_2 = _ssa_var_y_1;
                            switch (_polytemp_expr_2.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                                
                                    array_command_2 * cmd = (array_command_2 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_6;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_6);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_5<<<39063, 256>>>(dev_env, 10000000, _kernel_result_6);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_6;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_12 * cmd = (array_command_12 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_8;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_8, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_8);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_7<<<39063, 256>>>(dev_env, 10000000, _kernel_result_8);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_8;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_14 * cmd = (array_command_14 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_10;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_10, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_10);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_9<<<39063, 256>>>(dev_env, 10000000, _kernel_result_10);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_10;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_16 * cmd = (array_command_16 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_12;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_12, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_12);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_11<<<39063, 256>>>(dev_env, 10000000, _kernel_result_12);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_12;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_37 * cmd = (array_command_37 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_14;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_14, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_14);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_13<<<39063, 256>>>(dev_env, 10000000, _kernel_result_14, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_14;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 16: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_66 * cmd = (array_command_66 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_17;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_17, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_17);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_16<<<39063, 256>>>(dev_env, 10000000, _kernel_result_17, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_17;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 17: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_105 * cmd = (array_command_105 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_20;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_20, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_20);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_19<<<39063, 256>>>(dev_env, 10000000, _kernel_result_20, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_20;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_1;
                    })))));
                    _ssa_var_y_5 = _ssa_var_y_3;
                }
                else
                {
                    _ssa_var_y_4 = union_t(19, union_v_t::from_pointer((void *) new array_command_52(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_9;
                        {
                            union_t _polytemp_expr_10 = _ssa_var_y_1;
                            switch (_polytemp_expr_10.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                                
                                    array_command_2 * cmd = (array_command_2 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_74;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_74, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_74);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_73<<<39063, 256>>>(dev_env, 10000000, _kernel_result_74);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_74;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_12 * cmd = (array_command_12 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_76;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_76, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_76);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_75<<<39063, 256>>>(dev_env, 10000000, _kernel_result_76);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_76;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_14 * cmd = (array_command_14 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_78;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_78, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_78);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_77<<<39063, 256>>>(dev_env, 10000000, _kernel_result_78);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_78;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_16 * cmd = (array_command_16 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_80;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_80, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_80);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_79<<<39063, 256>>>(dev_env, 10000000, _kernel_result_80);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_80;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_37 * cmd = (array_command_37 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_82;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_82, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_82);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_81<<<39063, 256>>>(dev_env, 10000000, _kernel_result_82, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_82;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 16: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_66 * cmd = (array_command_66 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_85;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_85, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_85);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_84<<<39063, 256>>>(dev_env, 10000000, _kernel_result_85, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_85;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 17: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_105 * cmd = (array_command_105 *) _polytemp_expr_10.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_88;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_88, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_88);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_87<<<39063, 256>>>(dev_env, 10000000, _kernel_result_88, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_88;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_9;
                    })))));
                    _ssa_var_y_5 = _ssa_var_y_4;
                }
                _ssa_var_y_8 = _ssa_var_y_5;
            }
            else
            {
                _ssa_var_y_6 = union_t(20, union_v_t::from_pointer((void *) new array_command_89(NULL, new array_command_3(NULL, ({
                    variable_size_array_t _polytemp_result_17;
                    {
                        union_t _polytemp_expr_18 = _ssa_var_y_1;
                        switch (_polytemp_expr_18.class_id)
                        {
                            case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                            
                                array_command_2 * cmd = (array_command_2 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_142;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_142, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_142);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_141<<<39063, 256>>>(dev_env, 10000000, _kernel_result_142);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_142;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_12 * cmd = (array_command_12 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_144;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_144, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_144);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_143<<<39063, 256>>>(dev_env, 10000000, _kernel_result_144);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_144;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_14 * cmd = (array_command_14 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_146;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_146, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_146);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_145<<<39063, 256>>>(dev_env, 10000000, _kernel_result_146);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_146;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_16 * cmd = (array_command_16 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_148;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_148, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_148);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_147<<<39063, 256>>>(dev_env, 10000000, _kernel_result_148);
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_148;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_37 * cmd = (array_command_37 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_150;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_150, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_150);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_149<<<39063, 256>>>(dev_env, 10000000, _kernel_result_150, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_150;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 16: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_66 * cmd = (array_command_66 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_153;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_153, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_153);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_152<<<39063, 256>>>(dev_env, 10000000, _kernel_result_153, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_153;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 17: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_17 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_105 * cmd = (array_command_105 *) _polytemp_expr_18.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_156;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_156, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_156);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_155<<<39063, 256>>>(dev_env, 10000000, _kernel_result_156, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_156;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                        }
                    }
                    _polytemp_result_17;
                })))));
                _ssa_var_y_7 = ({
                    union_t _polytemp_result_25;
                    {
                        union_t _polytemp_expr_26 = _ssa_var_y_6;
                        switch (_polytemp_expr_26.class_id)
                        {
                            case 21: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_25 = union_t(22, union_v_t::from_pointer((void *) new array_command_10(NULL, (array_command_8 *) _polytemp_expr_26.value.pointer))); break;
                            case 20: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_25 = union_t(23, union_v_t::from_pointer((void *) new array_command_93(NULL, (array_command_89 *) _polytemp_expr_26.value.pointer))); break;
                        }
                    }
                    _polytemp_result_25;
                });
                _ssa_var_y_8 = _ssa_var_y_7;
            }
            _ssa_var_y_9 = ({
                union_t _polytemp_result_27;
                {
                    union_t _polytemp_expr_28 = _ssa_var_y_8;
                    switch (_polytemp_expr_28.class_id)
                    {
                        case 24: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(12, union_v_t::from_pointer((void *) new array_command_12(NULL, (array_command_4 *) _polytemp_expr_28.value.pointer))); break;
                        case 25: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(13, union_v_t::from_pointer((void *) new array_command_14(NULL, (array_command_6 *) _polytemp_expr_28.value.pointer))); break;
                        case 22: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(14, union_v_t::from_pointer((void *) new array_command_16(NULL, (array_command_10 *) _polytemp_expr_28.value.pointer))); break;
                        case 18: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(15, union_v_t::from_pointer((void *) new array_command_37(NULL, (array_command_23 *) _polytemp_expr_28.value.pointer))); break;
                        case 19: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(16, union_v_t::from_pointer((void *) new array_command_66(NULL, (array_command_52 *) _polytemp_expr_28.value.pointer))); break;
                        case 23: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_27 = union_t(17, union_v_t::from_pointer((void *) new array_command_105(NULL, (array_command_93 *) _polytemp_expr_28.value.pointer))); break;
                    }
                }
                _polytemp_result_27;
            });
            ({
                bool _polytemp_result_29;
                {
                    union_t _polytemp_expr_30 = _ssa_var_old_data_2;
                    switch (_polytemp_expr_30.class_id)
                    {
                        case 11: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_3 * cmd_to_free = (array_command_3 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_12 * cmd_to_free = (array_command_12 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_16 * cmd_to_free = (array_command_16 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_37 * cmd_to_free = (array_command_37 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 16: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_66 * cmd_to_free = (array_command_66 *) _polytemp_expr_30.value.pointer;
                        
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
                        case 17: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                            array_command_105 * cmd_to_free = (array_command_105 *) _polytemp_expr_30.value.pointer;
                        
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
                _polytemp_result_29;
            });
            _ssa_var_old_data_10 = _ssa_var_y_9;
            _ssa_var_y_1 = _ssa_var_y_9;
            _ssa_var_old_data_2 = _ssa_var_old_data_10;
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_31;
            {
                union_t _polytemp_expr_32 = _ssa_var_y_1;
                switch (_polytemp_expr_32.class_id)
                {
                    case 10: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_210;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_210, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_210);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_209<<<39063, 256>>>(dev_env, 10000000, _kernel_result_210);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_210;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_12 * cmd = (array_command_12 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_212;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_212, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_212);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_211<<<39063, 256>>>(dev_env, 10000000, _kernel_result_212);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_212;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_14 * cmd = (array_command_14 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_214;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_214, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_214);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_213<<<39063, 256>>>(dev_env, 10000000, _kernel_result_214);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_214;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_16 * cmd = (array_command_16 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_216;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_216, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_216);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_215<<<39063, 256>>>(dev_env, 10000000, _kernel_result_216);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_216;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_37 * cmd = (array_command_37 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_218;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_218, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_218);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_217<<<39063, 256>>>(dev_env, 10000000, _kernel_result_218, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_218;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 16: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_66 * cmd = (array_command_66 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_221;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_221, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_221);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_220<<<39063, 256>>>(dev_env, 10000000, _kernel_result_221, ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0.content))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_221;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 17: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [LVarReadNode: _ssa_var_y_8].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_105 * cmd = (array_command_105 *) _polytemp_expr_32.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_224;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_224, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_224);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_223<<<39063, 256>>>(dev_env, 10000000, _kernel_result_224, ((int *) ((int *) ((int *) ((int *) cmd->input_0->input_0->input_0->input_0.content)))));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_224;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                }
            }
            _polytemp_result_31;
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
