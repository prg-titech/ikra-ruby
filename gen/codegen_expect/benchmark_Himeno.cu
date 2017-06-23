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
    // Ikra::Symbolic::FixedSizeArrayInHostSectionCommand
    float *result;
    variable_size_array_t input_0;
    __host__ __device__ array_command_3(float *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayIndexCommand
    indexed_struct_3_lt_int_int_int_gt_t *result;
    __host__ __device__ array_command_5(indexed_struct_3_lt_int_int_int_gt_t *result = NULL) : result(result) { }
};
struct array_command_4 {
    // Ikra::Symbolic::ArrayStencilCommand
    float *result;
    array_command_3 *input_0;
    array_command_5 *input_1;
    __host__ __device__ array_command_4(float *result = NULL, array_command_3 *input_0 = NULL, array_command_5 *input_1 = NULL) : result(result), input_0(input_0), input_1(input_1) { }
};
struct environment_struct
{
    float * l4_param_a;
    float * l4_param_b;
    float * l4_param_c;
    float * l4_param_wrk;
    float * l4_param_bnd;
};

// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ float _block_k_2_(environment_t *_env_, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    
    int k;
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((65 - 1)))) / ((65 - 1)));
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
    {
        k = indices.field_2;
        {
            return (((((float) ((k * k))) / ((65 - 1)))) / ((65 - 1)));
        }
    }
}

#endif


__global__ void kernel_3(environment_t *_env_, int _num_threads_, float *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
}




__global__ void kernel_5(environment_t *_env_, int _num_threads_, float *_result_, float *_array_7_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_7_[_tid_];
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



// TODO: There should be a better to check if _block_k_4_ is already defined
#ifndef _block_k_4__func
#define _block_k_4__func
__device__ float _block_k_4_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    
    float * lex_param_bnd = _env_->l4_param_bnd;
    float * lex_param_wrk = _env_->l4_param_wrk;
    float * lex_param_c = _env_->l4_param_c;
    float * lex_param_b = _env_->l4_param_b;
    float * lex_param_a = _env_->l4_param_a;
    {
        return (values[0] + ((0.8 * ((((((((((((((((((((((((((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 0))] * values[1])) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 1))] * values[2])))) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 2))] * values[3])))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * values[16])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * values[17])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * values[18])))) + lex_param_wrk[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])) * lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 3))])) - values[0])) * lex_param_bnd[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])))));
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
        
        temp_stencil_13 = _block_k_4_(_env_, _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_11[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_11[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_13 = 0;
    }
        
        _result_[_tid_] = temp_stencil_13;
    }
}




__global__ void kernel_14(environment_t *_env_, int _num_threads_, float *_result_, float *_array_16_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_16_[_tid_];
    }
}




__global__ void kernel_19(environment_t *_env_, int _num_threads_, float *_result_, float *_array_21_)
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
__device__ float _block_k_4_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    
    float * lex_param_bnd = _env_->l4_param_bnd;
    float * lex_param_wrk = _env_->l4_param_wrk;
    float * lex_param_c = _env_->l4_param_c;
    float * lex_param_b = _env_->l4_param_b;
    float * lex_param_a = _env_->l4_param_a;
    {
        return (values[0] + ((0.8 * ((((((((((((((((((((((((((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 0))] * values[1])) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 1))] * values[2])))) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 2))] * values[3])))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * values[16])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * values[17])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * values[18])))) + lex_param_wrk[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])) * lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 3))])) - values[0])) * lex_param_bnd[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])))));
    }
}

#endif


__global__ void kernel_17(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_20)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_22;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_22 = _block_k_4_(_env_, _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_20[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_20[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_22 = 0;
    }
        
        _result_[_tid_] = temp_stencil_22;
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




__global__ void kernel_28(environment_t *_env_, int _num_threads_, float *_result_, float *_array_30_)
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
__device__ float _block_k_4_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    
    float * lex_param_bnd = _env_->l4_param_bnd;
    float * lex_param_wrk = _env_->l4_param_wrk;
    float * lex_param_c = _env_->l4_param_c;
    float * lex_param_b = _env_->l4_param_b;
    float * lex_param_a = _env_->l4_param_a;
    {
        return (values[0] + ((0.8 * ((((((((((((((((((((((((((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 0))] * values[1])) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 1))] * values[2])))) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 2))] * values[3])))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * values[16])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * values[17])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * values[18])))) + lex_param_wrk[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])) * lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 3))])) - values[0])) * lex_param_bnd[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])))));
    }
}

#endif


__global__ void kernel_26(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_29)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_31;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_31 = _block_k_4_(_env_, _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_29[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_29[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_31 = 0;
    }
        
        _result_[_tid_] = temp_stencil_31;
    }
}




__global__ void kernel_32(environment_t *_env_, int _num_threads_, float *_result_, float *_array_34_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_34_[_tid_];
    }
}




__global__ void kernel_37(environment_t *_env_, int _num_threads_, float *_result_, float *_array_39_)
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
__device__ float _block_k_4_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    
    float * lex_param_bnd = _env_->l4_param_bnd;
    float * lex_param_wrk = _env_->l4_param_wrk;
    float * lex_param_c = _env_->l4_param_c;
    float * lex_param_b = _env_->l4_param_b;
    float * lex_param_a = _env_->l4_param_a;
    {
        return (values[0] + ((0.8 * ((((((((((((((((((((((((((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 0))] * values[1])) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 1))] * values[2])))) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 2))] * values[3])))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * values[16])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * values[17])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * values[18])))) + lex_param_wrk[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])) * lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 3))])) - values[0])) * lex_param_bnd[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])))));
    }
}

#endif


__global__ void kernel_35(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_38)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_40;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_40 = _block_k_4_(_env_, _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_38[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_38[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_40 = 0;
    }
        
        _result_[_tid_] = temp_stencil_40;
    }
}




__global__ void kernel_41(environment_t *_env_, int _num_threads_, float *_result_, float *_array_43_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_43_[_tid_];
    }
}




__global__ void kernel_46(environment_t *_env_, int _num_threads_, float *_result_, float *_array_48_)
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
__device__ float _block_k_4_(environment_t *_env_, float _values_0, float _values_1, float _values_2, float _values_3, float _values_4, float _values_5, float _values_6, float _values_7, float _values_8, float _values_9, float _values_10, float _values_11, float _values_12, float _values_13, float _values_14, float _values_15, float _values_16, float _values_17, float _values_18, indexed_struct_3_lt_int_int_int_gt_t indices)
{
    
    // (Re)construct array from separately passed parameters
    float values[] = { _values_0, _values_1, _values_2, _values_3, _values_4, _values_5, _values_6, _values_7, _values_8, _values_9, _values_10, _values_11, _values_12, _values_13, _values_14, _values_15, _values_16, _values_17, _values_18 };
    
    float * lex_param_bnd = _env_->l4_param_bnd;
    float * lex_param_wrk = _env_->l4_param_wrk;
    float * lex_param_c = _env_->l4_param_c;
    float * lex_param_b = _env_->l4_param_b;
    float * lex_param_a = _env_->l4_param_a;
    {
        return (values[0] + ((0.8 * ((((((((((((((((((((((((((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 0))] * values[1])) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 1))] * values[2])))) + ((lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 2))] * values[3])))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * ((((((values[4] - values[5])) - values[6])) + values[7])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * ((((((values[8] - values[9])) - values[10])) + values[11])))))) + ((lex_param_b[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * ((((((values[12] - values[13])) - values[14])) + values[15])))))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 0))] * values[16])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 1))] * values[17])))) + ((lex_param_c[((((((((((((65 * 65)) * 3)) * indices.field_0)) + ((((65 * 3)) * indices.field_1)))) + ((3 * indices.field_2)))) + 2))] * values[18])))) + lex_param_wrk[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])) * lex_param_a[((((((((((((65 * 65)) * 4)) * indices.field_0)) + ((((65 * 4)) * indices.field_1)))) + ((4 * indices.field_2)))) + 3))])) - values[0])) * lex_param_bnd[((((((((65 * 65)) * indices.field_0)) + ((65 * indices.field_1)))) + indices.field_2))])))));
    }
}

#endif


__global__ void kernel_44(environment_t *_env_, int _num_threads_, float *_result_, float *_kernel_result_47)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    float temp_stencil_49;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 4225;
int temp_stencil_dim_1 = (_tid_ / 65) % 65;
int temp_stencil_dim_2 = (_tid_ / 1) % 65;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 129 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 1 < 65 && temp_stencil_dim_2 + -1 >= 0 && temp_stencil_dim_2 + 1 < 65)
    {
        // All value indices within bounds
        
        temp_stencil_49 = _block_k_4_(_env_, _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + -1) * 4225], _kernel_result_47[(temp_stencil_dim_2 + 0) * 1 + (temp_stencil_dim_1 + -1) * 65 + (temp_stencil_dim_0 + 0) * 4225], _kernel_result_47[(temp_stencil_dim_2 + -1) * 1 + (temp_stencil_dim_1 + 0) * 65 + (temp_stencil_dim_0 + 0) * 4225], ((indexed_struct_3_lt_int_int_int_gt_t) {_tid_ / 4225, (_tid_ / 65) % 65, (_tid_ / 1) % 65}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_49 = 0;
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
    array_command_2 * base = new array_command_2();
    union_t _ssa_var_old_old_data_3;
    array_command_4 * _ssa_var_next_p_6;
    union_t _ssa_var_old_data_5;
    union_t _ssa_var_old_old_data_4;
    int r;
    union_t _ssa_var_old_data_2;
    union_t _ssa_var_next_p_1;
    {
        _ssa_var_next_p_1 = union_t(10, union_v_t::from_pointer((void *) new array_command_3(NULL, ({
            // [Ikra::Symbolic::ArrayCombineCommand, size = 545025]
        
            array_command_2 * cmd = base;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            float * _kernel_result_2;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(float) * 545025)));
            program_result->device_allocations->push_back(_kernel_result_2);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_1<<<533, 1024>>>(dev_env, 545025, _kernel_result_2);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_2;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 545025);
        }))));
        _ssa_var_old_data_2 = union_t(11, union_v_t::from_pointer((void *) base));
        _ssa_var_old_old_data_3 = union_t(11, union_v_t::from_pointer((void *) base));
        for (r = 0; r <= (1000 - 1); r++)
        {
            _ssa_var_old_old_data_4 = _ssa_var_old_data_2;
            _ssa_var_old_data_5 = _ssa_var_next_p_1;
            _ssa_var_next_p_6 = new array_command_4(NULL, new array_command_3(NULL, ({
                variable_size_array_t _polytemp_result_1;
                {
                    union_t _polytemp_expr_2 = _ssa_var_next_p_1;
                    switch (_polytemp_expr_2.class_id)
                    {
                        case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 545025]
                        
                            array_command_3 * cmd = (array_command_3 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            float * _kernel_result_6;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_6);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_5<<<533, 1024>>>(dev_env, 545025, _kernel_result_6, ((float *) cmd->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_6;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 545025);
                        }); break;
                        case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_1 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_2.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            float * _kernel_result_11;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_11, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_11);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_10<<<533, 1024>>>(dev_env, 545025, _kernel_result_11, ((float *) ((float *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            float * _kernel_result_9;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_9, (sizeof(float) * 545025)));
                            program_result->device_allocations->push_back(_kernel_result_9);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_8<<<533, 1024>>>(dev_env, 545025, _kernel_result_9, _kernel_result_11);
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
                        
                            variable_size_array_t((void *) cmd->result, 545025);
                        }); break;
                    }
                }
                _polytemp_result_1;
            })));
            if (((r > 1)))
            {
                ({
                    bool _polytemp_result_9;
                    {
                        union_t _polytemp_expr_10 = _ssa_var_old_old_data_4;
                        switch (_polytemp_expr_10.class_id)
                        {
                            case 11: /* [Ikra::Symbolic::ArrayCombineCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
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
                            case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                array_command_3 * cmd_to_free = (array_command_3 *) _polytemp_expr_10.value.pointer;
                            
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
                            case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_9 = ({
                                array_command_4 * cmd_to_free = (array_command_4 *) _polytemp_expr_10.value.pointer;
                            
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
            }
            else
            {
            
            }
            _ssa_var_next_p_1 = union_t(12, union_v_t::from_pointer((void *) _ssa_var_next_p_6));
            _ssa_var_old_data_2 = _ssa_var_old_data_5;
            _ssa_var_old_old_data_3 = _ssa_var_old_old_data_4;
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_11;
            {
                union_t _polytemp_expr_12 = _ssa_var_next_p_1;
                switch (_polytemp_expr_12.class_id)
                {
                    case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 545025]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        float * _kernel_result_42;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_42, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_42);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_41<<<533, 1024>>>(dev_env, 545025, _kernel_result_42, ((float *) cmd->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_42;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 545025);
                    }); break;
                    case 12: /* [Ikra::Symbolic::ArrayStencilCommand, size = 545025] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_11 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 545025]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_next_p_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>]], [ArrayNode: [<0>, <1>, <0>]], [ArrayNode: [<0>, <0>, <1>]], [ArrayNode: [<1>, <1>, <0>]], [ArrayNode: [<1>, <-1>, <0>]], [ArrayNode: [<-1>, <1>, <0>]], [ArrayNode: [<-1>, <-1>, <0>]], [ArrayNode: [<0>, <1>, <1>]], [ArrayNode: [<0>, <-1>, <1>]], [ArrayNode: [<0>, <1>, <-1>]], [ArrayNode: [<0>, <-1>, <-1>]], [ArrayNode: [<1>, <0>, <1>]], [ArrayNode: [<-1>, <0>, <1>]], [ArrayNode: [<1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <-1>]], [ArrayNode: [<-1>, <0>, <0>]], [ArrayNode: [<0>, <-1>, <0>]], [ArrayNode: [<0>, <0>, <-1>]]]]; <0>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_4 * cmd = (array_command_4 *) _polytemp_expr_12.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        float * _kernel_result_47;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_47, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_47);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_46<<<533, 1024>>>(dev_env, 545025, _kernel_result_47, ((float *) ((float *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        float * _kernel_result_45;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_45, (sizeof(float) * 545025)));
                        program_result->device_allocations->push_back(_kernel_result_45);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_44<<<533, 1024>>>(dev_env, 545025, _kernel_result_45, _kernel_result_47);
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
                    
                        variable_size_array_t((void *) cmd->result, 545025);
                    }); break;
                }
            }
            _polytemp_result_11;
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
    
    void * temp_ptr_l4_param_a = host_env->l4_param_a;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->l4_param_a, 8720400));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(host_env->l4_param_a, temp_ptr_l4_param_a, 8720400, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);

    void * temp_ptr_l4_param_b = host_env->l4_param_b;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->l4_param_b, 6540300));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(host_env->l4_param_b, temp_ptr_l4_param_b, 6540300, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);

    void * temp_ptr_l4_param_c = host_env->l4_param_c;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->l4_param_c, 6540300));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(host_env->l4_param_c, temp_ptr_l4_param_c, 6540300, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);

    void * temp_ptr_l4_param_wrk = host_env->l4_param_wrk;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->l4_param_wrk, 2180100));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(host_env->l4_param_wrk, temp_ptr_l4_param_wrk, 2180100, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);

    void * temp_ptr_l4_param_bnd = host_env->l4_param_bnd;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->l4_param_bnd, 2180100));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(host_env->l4_param_bnd, temp_ptr_l4_param_bnd, 2180100, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);
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
