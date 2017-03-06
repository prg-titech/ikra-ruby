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
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_117 = ((indices.field_1 % 4));
        (_temp_var_117 == 0 ? indices.field_0 : (_temp_var_117 == 1 ? indices.field_1 : (_temp_var_117 == 2 ? indices.field_2 : (_temp_var_117 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_197(environment_t *_env_, int _num_threads_, int *_result_)
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
        return (((indices.field_0 + indices.field_1)) % ((((indices.field_3 + ({ int _temp_var_118 = ((indices.field_1 % 4));
        (_temp_var_118 == 0 ? indices.field_0 : (_temp_var_118 == 1 ? indices.field_1 : (_temp_var_118 == 2 ? indices.field_2 : (_temp_var_118 == 3 ? indices.field_3 : NULL)))); }))) + 7)));
    }
}

#endif


__global__ void kernel_199(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
}




__global__ void kernel_201(environment_t *_env_, int _num_threads_, int *_result_, int *_array_203_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_203_[_tid_];
    }
}




__global__ void kernel_206(environment_t *_env_, int _num_threads_, int *_result_, int *_array_208_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_208_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_121 = ((({ int _temp_var_122 = ((({ int _temp_var_123 = ((values[2] % 4));
        (_temp_var_123 == 0 ? indices.field_0 : (_temp_var_123 == 1 ? indices.field_1 : (_temp_var_123 == 2 ? indices.field_2 : (_temp_var_123 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_122 == 0 ? indices.field_0 : (_temp_var_122 == 1 ? indices.field_1 : (_temp_var_122 == 2 ? indices.field_2 : (_temp_var_122 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_121 == 0 ? indices.field_0 : (_temp_var_121 == 1 ? indices.field_1 : (_temp_var_121 == 2 ? indices.field_2 : (_temp_var_121 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_204(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_207)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_209;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_209 = _block_k_4_(_env_, _kernel_result_207[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_207[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_207[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_207[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_209 = 37;
    }
        
        _result_[_tid_] = temp_stencil_209;
    }
}




__global__ void kernel_210(environment_t *_env_, int _num_threads_, int *_result_, int *_array_212_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_212_[_tid_];
    }
}




__global__ void kernel_215(environment_t *_env_, int _num_threads_, int *_result_, int *_array_217_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_217_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_126 = ((({ int _temp_var_127 = ((({ int _temp_var_128 = ((values[2] % 4));
        (_temp_var_128 == 0 ? indices.field_0 : (_temp_var_128 == 1 ? indices.field_1 : (_temp_var_128 == 2 ? indices.field_2 : (_temp_var_128 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_127 == 0 ? indices.field_0 : (_temp_var_127 == 1 ? indices.field_1 : (_temp_var_127 == 2 ? indices.field_2 : (_temp_var_127 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_126 == 0 ? indices.field_0 : (_temp_var_126 == 1 ? indices.field_1 : (_temp_var_126 == 2 ? indices.field_2 : (_temp_var_126 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_213(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_216)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_218;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_218 = _block_k_4_(_env_, _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_216[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_218 = 37;
    }
        
        _result_[_tid_] = temp_stencil_218;
    }
}




__global__ void kernel_219(environment_t *_env_, int _num_threads_, int *_result_, int *_array_221_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_221_[_tid_];
    }
}




__global__ void kernel_224(environment_t *_env_, int _num_threads_, int *_result_, int *_array_226_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_226_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_131 = ((({ int _temp_var_132 = ((({ int _temp_var_133 = ((values[2] % 4));
        (_temp_var_133 == 0 ? indices.field_0 : (_temp_var_133 == 1 ? indices.field_1 : (_temp_var_133 == 2 ? indices.field_2 : (_temp_var_133 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_132 == 0 ? indices.field_0 : (_temp_var_132 == 1 ? indices.field_1 : (_temp_var_132 == 2 ? indices.field_2 : (_temp_var_132 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_131 == 0 ? indices.field_0 : (_temp_var_131 == 1 ? indices.field_1 : (_temp_var_131 == 2 ? indices.field_2 : (_temp_var_131 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_222(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_225)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_227;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_227 = _block_k_4_(_env_, _kernel_result_225[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_225[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_225[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_225[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_227 = 37;
    }
        
        _result_[_tid_] = temp_stencil_227;
    }
}




__global__ void kernel_228(environment_t *_env_, int _num_threads_, int *_result_, int *_array_230_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_230_[_tid_];
    }
}




__global__ void kernel_233(environment_t *_env_, int _num_threads_, int *_result_, int *_array_235_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_235_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_136 = ((({ int _temp_var_137 = ((({ int _temp_var_138 = ((values[2] % 4));
        (_temp_var_138 == 0 ? indices.field_0 : (_temp_var_138 == 1 ? indices.field_1 : (_temp_var_138 == 2 ? indices.field_2 : (_temp_var_138 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_137 == 0 ? indices.field_0 : (_temp_var_137 == 1 ? indices.field_1 : (_temp_var_137 == 2 ? indices.field_2 : (_temp_var_137 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_136 == 0 ? indices.field_0 : (_temp_var_136 == 1 ? indices.field_1 : (_temp_var_136 == 2 ? indices.field_2 : (_temp_var_136 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_231(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_234)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_236;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_236 = _block_k_4_(_env_, _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_234[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_236 = 37;
    }
        
        _result_[_tid_] = temp_stencil_236;
    }
}




__global__ void kernel_237(environment_t *_env_, int _num_threads_, int *_result_, int *_array_239_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_239_[_tid_];
    }
}




__global__ void kernel_242(environment_t *_env_, int _num_threads_, int *_result_, int *_array_244_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _array_244_[_tid_];
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
        return (((((((values[0] % 938)) + ((values[1] / 97)))) % 97717)) + ((((({ int _temp_var_143 = ((({ int _temp_var_144 = ((({ int _temp_var_145 = ((values[2] % 4));
        (_temp_var_145 == 0 ? indices.field_0 : (_temp_var_145 == 1 ? indices.field_1 : (_temp_var_145 == 2 ? indices.field_2 : (_temp_var_145 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_144 == 0 ? indices.field_0 : (_temp_var_144 == 1 ? indices.field_1 : (_temp_var_144 == 2 ? indices.field_2 : (_temp_var_144 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_143 == 0 ? indices.field_0 : (_temp_var_143 == 1 ? indices.field_1 : (_temp_var_143 == 2 ? indices.field_2 : (_temp_var_143 == 3 ? indices.field_3 : NULL)))); }) * ((values[3] % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_240(environment_t *_env_, int _num_threads_, int *_result_, int *_kernel_result_243)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {
    int temp_stencil_245;

    // Indices for all dimensions
    int temp_stencil_dim_0 = _tid_ / 500000;
int temp_stencil_dim_1 = (_tid_ / 1000) % 500;
int temp_stencil_dim_2 = (_tid_ / 2) % 500;
int temp_stencil_dim_3 = (_tid_ / 1) % 2;

    if (temp_stencil_dim_0 + -1 >= 0 && temp_stencil_dim_0 + 1 < 20 && temp_stencil_dim_1 + -1 >= 0 && temp_stencil_dim_1 + 0 < 500 && temp_stencil_dim_2 + 0 >= 0 && temp_stencil_dim_2 + 0 < 500 && temp_stencil_dim_3 + 0 >= 0 && temp_stencil_dim_3 + 0 < 2)
    {
        // All value indices within bounds
        
        temp_stencil_245 = _block_k_4_(_env_, _kernel_result_243[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + -1) * 500000], _kernel_result_243[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 0) * 500000], _kernel_result_243[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + 0) * 1000 + (temp_stencil_dim_0 + 1) * 500000], _kernel_result_243[(temp_stencil_dim_3 + 0) * 1 + (temp_stencil_dim_2 + 0) * 2 + (temp_stencil_dim_1 + -1) * 1000 + (temp_stencil_dim_0 + -1) * 500000], ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
    }
    else
    {
        // At least one index is out of bounds
        temp_stencil_245 = 37;
    }
        
        _result_[_tid_] = temp_stencil_245;
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
            // [Ikra::Symbolic::ArrayCombineCommand, size = 10000000]
        
            array_command_2 * cmd = x;
        
            if (cmd->result == 0) {
                    timeStartMeasure();
            int * _kernel_result_198;
            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_198, (sizeof(int) * 10000000)));
            program_result->device_allocations->push_back(_kernel_result_198);
            timeReportMeasure(program_result, allocate_memory);
            timeStartMeasure();
            kernel_197<<<39063, 256>>>(dev_env, 10000000, _kernel_result_198);
            checkErrorReturn(program_result, cudaPeekAtLastError());
            checkErrorReturn(program_result, cudaThreadSynchronize());
            timeReportMeasure(program_result, kernel);
                cmd->result = _kernel_result_198;
        
                
            }
        
            variable_size_array_t((void *) cmd->result, 10000000);
        }))));
        _ssa_var_old_data_2 = union_t(19, union_v_t::from_pointer((void *) x));
        _ssa_var_old_old_data_3 = union_t(19, union_v_t::from_pointer((void *) x));
        for (r = 0; r <= (200 - 1); r++)
        {
            _ssa_var_old_old_data_4 = _ssa_var_old_data_2;
            _ssa_var_old_data_5 = _ssa_var_y_1;
            _ssa_var_y_6 = new array_command_4(NULL, new array_command_3(NULL, ({
                variable_size_array_t _polytemp_result_119;
                {
                    union_t _polytemp_expr_120 = _ssa_var_y_1;
                    switch (_polytemp_expr_120.class_id)
                    {
                        case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_119 = ({
                            // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                        
                            array_command_3 * cmd = (array_command_3 *) _polytemp_expr_120.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_202;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_202, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_202);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_201<<<39063, 256>>>(dev_env, 10000000, _kernel_result_202, ((int *) cmd->input_0.content));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_202;
                        
                                
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                        case 20: /* [Ikra::Symbolic::ArrayStencilCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_119 = ({
                            // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                        
                            array_command_4 * cmd = (array_command_4 *) _polytemp_expr_120.value.pointer;
                        
                            if (cmd->result == 0) {
                                    timeStartMeasure();
                            int * _kernel_result_207;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_207, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_207);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_206<<<39063, 256>>>(dev_env, 10000000, _kernel_result_207, ((int *) ((int *) cmd->input_0->input_0.content)));
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);    timeStartMeasure();
                            int * _kernel_result_205;
                            checkErrorReturn(program_result, cudaMalloc(&_kernel_result_205, (sizeof(int) * 10000000)));
                            program_result->device_allocations->push_back(_kernel_result_205);
                            timeReportMeasure(program_result, allocate_memory);
                            timeStartMeasure();
                            kernel_204<<<39063, 256>>>(dev_env, 10000000, _kernel_result_205, _kernel_result_207);
                            checkErrorReturn(program_result, cudaPeekAtLastError());
                            checkErrorReturn(program_result, cudaThreadSynchronize());
                            timeReportMeasure(program_result, kernel);
                                cmd->result = _kernel_result_205;
                        
                                    timeStartMeasure();
                        
                            if (_kernel_result_207 != cmd->result) {
                                // Don't free memory if it is the result. There is already a similar check in
                                // program_builder (free all except for last). However, this check is not sufficient in
                                // case the same array is reused!
                        
                                checkErrorReturn(program_result, cudaFree(_kernel_result_207));
                                // Remove from list of allocations
                                program_result->device_allocations->erase(
                                    std::remove(
                                        program_result->device_allocations->begin(),
                                        program_result->device_allocations->end(),
                                        _kernel_result_207),
                                    program_result->device_allocations->end());
                            }
                        
                            timeReportMeasure(program_result, free_memory);
                        
                            }
                        
                            variable_size_array_t((void *) cmd->result, 10000000);
                        }); break;
                    }
                }
                _polytemp_result_119;
            })));
            if (((r > 1)))
            {
                ({
                    bool _polytemp_result_139;
                    {
                        union_t _polytemp_expr_140 = _ssa_var_old_old_data_4;
                        switch (_polytemp_expr_140.class_id)
                        {
                            case 19: /* [Ikra::Symbolic::ArrayCombineCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_139 = ({
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
                            case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_139 = ({
                                array_command_3 * cmd_to_free = (array_command_3 *) _polytemp_expr_140.value.pointer;
                            
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
                            case 20: /* [Ikra::Symbolic::ArrayStencilCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_139 = ({
                                array_command_4 * cmd_to_free = (array_command_4 *) _polytemp_expr_140.value.pointer;
                            
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
            }
            else
            {
            
            }
            _ssa_var_y_1 = union_t(20, union_v_t::from_pointer((void *) _ssa_var_y_6));
            _ssa_var_old_data_2 = _ssa_var_old_data_5;
            _ssa_var_old_old_data_3 = _ssa_var_old_old_data_4;
        }
        r--;
        return ({
            variable_size_array_t _polytemp_result_141;
            {
                union_t _polytemp_expr_142 = _ssa_var_y_1;
                switch (_polytemp_expr_142.class_id)
                {
                    case 10: /* [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_141 = ({
                        // [Ikra::Symbolic::FixedSizeArrayInHostSectionCommand, size = 10000000]
                    
                        array_command_3 * cmd = (array_command_3 *) _polytemp_expr_142.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_238;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_238, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_238);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_237<<<39063, 256>>>(dev_env, 10000000, _kernel_result_238, ((int *) cmd->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_238;
                    
                            
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                    case 20: /* [Ikra::Symbolic::ArrayStencilCommand, size = 10000000] (Ikra::Symbolic::ArrayCommand) */ _polytemp_result_141 = ({
                        // [Ikra::Symbolic::ArrayStencilCommand, size = 10000000]: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_y_1].__call__()].to_command()].pstencil([ArrayNode: [[ArrayNode: [<-1>, <0>, <0>, <0>]], [ArrayNode: [<0>, <0>, <0>, <0>]], [ArrayNode: [<1>, <0>, <0>, <0>]], [ArrayNode: [<-1>, <-1>, <0>, <0>]]]]; <37>; [HashNode: {<:with_index> => [BeginNode: {<true>}]}])]
                    
                        array_command_4 * cmd = (array_command_4 *) _polytemp_expr_142.value.pointer;
                    
                        if (cmd->result == 0) {
                                timeStartMeasure();
                        int * _kernel_result_243;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_243, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_243);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_242<<<39063, 256>>>(dev_env, 10000000, _kernel_result_243, ((int *) ((int *) cmd->input_0->input_0.content)));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);    timeStartMeasure();
                        int * _kernel_result_241;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_241, (sizeof(int) * 10000000)));
                        program_result->device_allocations->push_back(_kernel_result_241);
                        timeReportMeasure(program_result, allocate_memory);
                        timeStartMeasure();
                        kernel_240<<<39063, 256>>>(dev_env, 10000000, _kernel_result_241, _kernel_result_243);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                        timeReportMeasure(program_result, kernel);
                            cmd->result = _kernel_result_241;
                    
                                timeStartMeasure();
                    
                        if (_kernel_result_243 != cmd->result) {
                            // Don't free memory if it is the result. There is already a similar check in
                            // program_builder (free all except for last). However, this check is not sufficient in
                            // case the same array is reused!
                    
                            checkErrorReturn(program_result, cudaFree(_kernel_result_243));
                            // Remove from list of allocations
                            program_result->device_allocations->erase(
                                std::remove(
                                    program_result->device_allocations->begin(),
                                    program_result->device_allocations->end(),
                                    _kernel_result_243),
                                program_result->device_allocations->end());
                        }
                    
                        timeReportMeasure(program_result, free_memory);
                    
                        }
                    
                        variable_size_array_t((void *) cmd->result, 10000000);
                    }); break;
                }
            }
            _polytemp_result_141;
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
