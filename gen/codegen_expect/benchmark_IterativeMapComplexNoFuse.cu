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
struct array_command_6 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_4_lt_int_int_int_int_gt_t *result;
    __host__ __device__ array_command_6(indexed_struct_4_lt_int_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_17 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_6 *input_1;
    __host__ __device__ array_command_17(int *result = NULL, array_command_3 *input_0 = NULL, array_command_6 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_11 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_6 *input_1;
    __host__ __device__ array_command_11(int *result = NULL, array_command_3 *input_0 = NULL, array_command_6 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_6 *input_1;
    __host__ __device__ array_command_5(int *result = NULL, array_command_3 *input_0 = NULL, array_command_6 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_8 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_6 *input_1;
    __host__ __device__ array_command_8(int *result = NULL, array_command_3 *input_0 = NULL, array_command_6 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct array_command_14 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_3 *input_0;
    array_command_6 *input_1;
    __host__ __device__ array_command_14(int *result = NULL, array_command_3 *input_0 = NULL, array_command_6 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
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




__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_, int *_array_7_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_7_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_8(environment_t *_env_, int _num_threads_, int *_result_, int *_array_10_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_10_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_11(environment_t *_env_, int _num_threads_, int *_result_, int *_array_13_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_13_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_14(environment_t *_env_, int _num_threads_, int *_result_, int *_array_16_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_16_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_17(environment_t *_env_, int _num_threads_, int *_result_, int *_array_19_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_19_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_20(environment_t *_env_, int _num_threads_, int *_result_, int *_array_22_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_22_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_26(environment_t *_env_, int _num_threads_, int *_result_, int *_array_28_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_28_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_29(environment_t *_env_, int _num_threads_, int *_result_, int *_array_31_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_31_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_32(environment_t *_env_, int _num_threads_, int *_result_, int *_array_34_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_34_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_35(environment_t *_env_, int _num_threads_, int *_result_, int *_array_37_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_37_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_38(environment_t *_env_, int _num_threads_, int *_result_, int *_array_40_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_40_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_44(environment_t *_env_, int _num_threads_, int *_result_, int *_array_46_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_46_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_47(environment_t *_env_, int _num_threads_, int *_result_, int *_array_49_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_49_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
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

        
        _result_[_tid_] = _block_k_17_(_env_, _array_52_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_53(environment_t *_env_, int _num_threads_, int *_result_, int *_array_55_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_55_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_56(environment_t *_env_, int _num_threads_, int *_result_, int *_array_58_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_58_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_59(environment_t *_env_, int _num_threads_, int *_result_, int *_array_61_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_61_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_62(environment_t *_env_, int _num_threads_, int *_result_, int *_array_64_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_64_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_65(environment_t *_env_, int _num_threads_, int *_result_, int *_array_67_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_67_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_68(environment_t *_env_, int _num_threads_, int *_result_, int *_array_70_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_70_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_71(environment_t *_env_, int _num_threads_, int *_result_, int *_array_73_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_73_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_74(environment_t *_env_, int _num_threads_, int *_result_, int *_array_76_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_76_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_77(environment_t *_env_, int _num_threads_, int *_result_, int *_array_79_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_79_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_80(environment_t *_env_, int _num_threads_, int *_result_, int *_array_82_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_82_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_83(environment_t *_env_, int _num_threads_, int *_result_, int *_array_85_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_85_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif


__global__ void kernel_86(environment_t *_env_, int _num_threads_, int *_result_, int *_array_88_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_11_(_env_, _array_88_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_89(environment_t *_env_, int _num_threads_, int *_result_, int *_array_91_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_5_(_env_, _array_91_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_92(environment_t *_env_, int _num_threads_, int *_result_, int *_array_94_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_94_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_95(environment_t *_env_, int _num_threads_, int *_result_, int *_array_97_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _array_97_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_98(environment_t *_env_, int _num_threads_, int *_result_, int *_array_100_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_5_(_env_, _array_100_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_101(environment_t *_env_, int _num_threads_, int *_result_, int *_array_103_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_103_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_104(environment_t *_env_, int _num_threads_, int *_result_, int *_array_106_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _array_106_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_107(environment_t *_env_, int _num_threads_, int *_result_, int *_array_109_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_5_(_env_, _array_109_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_110(environment_t *_env_, int _num_threads_, int *_result_, int *_array_112_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_112_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_113(environment_t *_env_, int _num_threads_, int *_result_, int *_array_115_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _array_115_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif


__global__ void kernel_116(environment_t *_env_, int _num_threads_, int *_result_, int *_array_118_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_5_(_env_, _array_118_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_8_ is already defined
#ifndef _block_k_8__func
#define _block_k_8__func
__device__ int _block_k_8_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif


__global__ void kernel_119(environment_t *_env_, int _num_threads_, int *_result_, int *_array_121_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_8_(_env_, _array_121_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}



// TODO: There should be a better to check if _block_k_14_ is already defined
#ifndef _block_k_14__func
#define _block_k_14__func
__device__ int _block_k_14_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif


__global__ void kernel_122(environment_t *_env_, int _num_threads_, int *_result_, int *_array_124_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_14_(_env_, _array_124_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_125(environment_t *_env_, int _num_threads_, int *_result_, int *_array_127_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_127_[_tid_];
    }
}



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_128(environment_t *_env_, int _num_threads_, int *_result_, int *_array_130_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_17_(_env_, _array_130_[_tid_], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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
    array_command_17 * _ssa_var_old_data_16;
    array_command_17 * _ssa_var_y_15;
    array_command_14 * _ssa_var_old_data_12;
    array_command_14 * _ssa_var_y_11;
    array_command_11 * _ssa_var_old_data_10;
    array_command_11 * _ssa_var_y_9;
    union_t _ssa_var_old_data_14;
    union_t _ssa_var_y_13;
    array_command_8 * _ssa_var_old_data_6;
    array_command_8 * _ssa_var_y_5;
    union_t _ssa_var_old_data_8;
    union_t _ssa_var_y_7;
    array_command_5 * _ssa_var_old_data_4;
    array_command_5 * _ssa_var_y_3;
    int r;
    union_t _ssa_var_old_data_2;
    union_t _ssa_var_y_1;
    {
        _ssa_var_y_1 = union_t(10, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
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
        _ssa_var_old_data_2 = union_t(11, union_v_t::from_pointer((void *) x));
        for (r = 0; r <= (200 - 1); r++)
        {
            if (((((r % 2)) == 0)))
            {
                if (((((r % 3)) == 0)))
                {
                    _ssa_var_y_3 = new array_command_5(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_1;
                        {
                            union_t _polytemp_expr_2 = _ssa_var_y_1;
                            switch (_polytemp_expr_2.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                                
                                    array_command_3 * cmd = (array_command_3 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_6;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_6);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_5<<<39063, 256>>>(dev_env, 10000000, _kernel_result_6, ((int *) cmd->input_0.content));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_6;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_13].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_17 * cmd = (array_command_17 *) _polytemp_expr_2.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_9;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_9, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_9);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_8<<<39063, 256>>>(dev_env, 10000000, _kernel_result_9, ((int *) ((int *) cmd->input_0->input_0.content)));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_9;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_1;
                    })));
                    ({
                        bool _polytemp_result_9;
                        {
                            union_t _polytemp_expr_10 = _ssa_var_old_data_2;
                            switch (_polytemp_expr_10.class_id)
                            {
                                case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_10.value.pointer;
                                
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
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                    array_command_17 * cmd_to_free = (array_command_17 *) _polytemp_expr_10.value.pointer;
                                
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
                        _polytemp_result_9;
                    });
                    _ssa_var_old_data_4 = _ssa_var_y_3;
                    _ssa_var_y_7 = union_t(13, union_v_t::from_pointer((void *) _ssa_var_y_3));
                    _ssa_var_old_data_8 = union_t(13, union_v_t::from_pointer((void *) _ssa_var_old_data_4));
                }
                else
                {
                    _ssa_var_y_5 = new array_command_8(NULL, new array_command_3(NULL, ({
                        variable_size_array_t _polytemp_result_11;
                        {
                            union_t _polytemp_expr_12 = _ssa_var_y_1;
                            switch (_polytemp_expr_12.class_id)
                            {
                                case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                                    // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                                
                                    array_command_3 * cmd = (array_command_3 *) _polytemp_expr_12.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_30;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_30, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_30);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_29<<<39063, 256>>>(dev_env, 10000000, _kernel_result_30, ((int *) cmd->input_0.content));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_30;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_13].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                                
                                    array_command_17 * cmd = (array_command_17 *) _polytemp_expr_12.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            timeStartMeasure();
                                    int * _kernel_result_33;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_33, (sizeof(int) * 10000000)));
                                    program_result->device_allocations->push_back(_kernel_result_33);
                                    timeReportMeasure(program_result, allocate_memory);
                                    timeStartMeasure();
                                    kernel_32<<<39063, 256>>>(dev_env, 10000000, _kernel_result_33, ((int *) ((int *) cmd->input_0->input_0.content)));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                    timeReportMeasure(program_result, kernel);
                                        cmd->result = _kernel_result_33;
                                
                                        
                                    }
                                
                                    variable_size_array_t((void *) cmd->result, 10000000);
                                }); break;
                            }
                        }
                        _polytemp_result_11;
                    })));
                    ({
                        bool _polytemp_result_19;
                        {
                            union_t _polytemp_expr_20 = _ssa_var_old_data_2;
                            switch (_polytemp_expr_20.class_id)
                            {
                                case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_19 = ({
                                    array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_20.value.pointer;
                                
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
                                case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_19 = ({
                                    array_command_17 * cmd_to_free = (array_command_17 *) _polytemp_expr_20.value.pointer;
                                
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
                        _polytemp_result_19;
                    });
                    _ssa_var_old_data_6 = _ssa_var_y_5;
                    _ssa_var_y_7 = union_t(14, union_v_t::from_pointer((void *) _ssa_var_y_5));
                    _ssa_var_old_data_8 = union_t(14, union_v_t::from_pointer((void *) _ssa_var_old_data_6));
                }
                _ssa_var_y_13 = _ssa_var_y_7;
                _ssa_var_old_data_14 = _ssa_var_old_data_8;
            }
            else
            {
                _ssa_var_y_9 = new array_command_11(NULL, new array_command_3(NULL, ({
                    variable_size_array_t _polytemp_result_21;
                    {
                        union_t _polytemp_expr_22 = _ssa_var_y_1;
                        switch (_polytemp_expr_22.class_id)
                        {
                            case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_21 = ({
                                // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                            
                                array_command_3 * cmd = (array_command_3 *) _polytemp_expr_22.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_54;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_54, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_54);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_53<<<39063, 256>>>(dev_env, 10000000, _kernel_result_54, ((int *) cmd->input_0.content));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_54;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                            case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_21 = ({
                                // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_13].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                            
                                array_command_17 * cmd = (array_command_17 *) _polytemp_expr_22.value.pointer;
                            
                                if (cmd->result == 0) {
                                        timeStartMeasure();
                                int * _kernel_result_57;
                                checkErrorReturn(program_result, cudaMalloc(&_kernel_result_57, (sizeof(int) * 10000000)));
                                program_result->device_allocations->push_back(_kernel_result_57);
                                timeReportMeasure(program_result, allocate_memory);
                                timeStartMeasure();
                                kernel_56<<<39063, 256>>>(dev_env, 10000000, _kernel_result_57, ((int *) ((int *) cmd->input_0->input_0.content)));
                                checkErrorReturn(program_result, cudaPeekAtLastError());
                                checkErrorReturn(program_result, cudaThreadSynchronize());
                                timeReportMeasure(program_result, kernel);
                                    cmd->result = _kernel_result_57;
                            
                                    
                                }
                            
                                variable_size_array_t((void *) cmd->result, 10000000);
                            }); break;
                        }
                    }
                    _polytemp_result_21;
                })));
                ({
                    bool _polytemp_result_29;
                    {
                        union_t _polytemp_expr_30 = _ssa_var_old_data_2;
                        switch (_polytemp_expr_30.class_id)
                        {
                            case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_29 = ({
                                array_command_2 * cmd_to_free = (array_command_2 *) _polytemp_expr_30.value.pointer;
                            
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
                                array_command_17 * cmd_to_free = (array_command_17 *) _polytemp_expr_30.value.pointer;
                            
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
                _ssa_var_y_11 = new array_command_14(NULL, new array_command_3(NULL, ({
                    // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                
                    array_command_11 * cmd = _ssa_var_y_9;
                
                    if (cmd->result == 0) {
                            timeStartMeasure();
                    int * _kernel_result_78;
                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_78, (sizeof(int) * 10000000)));
                    program_result->device_allocations->push_back(_kernel_result_78);
                    timeReportMeasure(program_result, allocate_memory);
                    timeStartMeasure();
                    kernel_77<<<39063, 256>>>(dev_env, 10000000, _kernel_result_78, ((int *) ((int *) cmd->input_0->input_0.content)));
                    checkErrorReturn(program_result, cudaPeekAtLastError());
                    checkErrorReturn(program_result, cudaThreadSynchronize());
                    timeReportMeasure(program_result, kernel);
                        cmd->result = _kernel_result_78;
                
                        
                    }
                
                    variable_size_array_t((void *) cmd->result, 10000000);
                })));
                ({
                    array_command_11 * cmd_to_free = _ssa_var_old_data_10;
                
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
                });
                _ssa_var_old_data_12 = _ssa_var_y_11;
                _ssa_var_y_13 = union_t(15, union_v_t::from_pointer((void *) _ssa_var_y_11));
                _ssa_var_old_data_14 = union_t(15, union_v_t::from_pointer((void *) _ssa_var_old_data_12));
            }
            _ssa_var_y_15 = new array_command_17(NULL, new array_command_3(NULL, ({
                variable_size_array_t _polytemp_result_31;
                {
                    union_t _polytemp_expr_32 = _ssa_var_y_13;
                    switch (_polytemp_expr_32.class_id)
                    {
                        case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_5 * cmd = (array_command_5 *) _polytemp_expr_32.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_90;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_90, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_90);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_89<<<39063, 256>>>(dev_env, 10000000, _kernel_result_90, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_90;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_8 * cmd = (array_command_8 *) _polytemp_expr_32.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_93;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_93, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_93);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_92<<<39063, 256>>>(dev_env, 10000000, _kernel_result_93, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_93;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_31 = ({
                            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_9].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_14 * cmd = (array_command_14 *) _polytemp_expr_32.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_96;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_96, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_96);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_95<<<39063, 256>>>(dev_env, 10000000, _kernel_result_96, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_96;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                    }
                }
                _polytemp_result_31;
            })));
            ({
                bool _polytemp_result_39;
                {
                    union_t _polytemp_expr_40 = _ssa_var_old_data_14;
                    switch (_polytemp_expr_40.class_id)
                    {
                        case 13: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_39 = ({
                            array_command_5 * cmd_to_free = (array_command_5 *) _polytemp_expr_40.value.pointer;
                        
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
                        case 14: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_39 = ({
                            array_command_8 * cmd_to_free = (array_command_8 *) _polytemp_expr_40.value.pointer;
                        
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
                        case 15: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_39 = ({
                            array_command_14 * cmd_to_free = (array_command_14 *) _polytemp_expr_40.value.pointer;
                        
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
                _polytemp_result_39;
            });
            _ssa_var_old_data_16 = _ssa_var_y_15;
            _ssa_var_y_1 = union_t(12, union_v_t::from_pointer((void *) _ssa_var_y_15));
            _ssa_var_old_data_2 = union_t(12, union_v_t::from_pointer((void *) _ssa_var_old_data_16));
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_41;
            {
                union_t _polytemp_expr_42 = _ssa_var_y_1;
                switch (_polytemp_expr_42.class_id)
                {
                    case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_41 = ({
                        // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_42.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_126;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_126, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_126);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_125<<<39063, 256>>>(dev_env, 10000000, _kernel_result_126, ((int *) cmd->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_126;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_41 = ({
                        // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_13].__call__()].to_command()].pmap([HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_17 * cmd = (array_command_17 *) _polytemp_expr_42.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_129;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_129, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_129);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_128<<<39063, 256>>>(dev_env, 10000000, _kernel_result_129, ((int *) ((int *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_129;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                }
            }
            _polytemp_result_41;
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
