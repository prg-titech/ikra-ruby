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



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_2 = ((({ int _temp_var_3 = ((({ int _temp_var_4 = ((i % 4));
        (_temp_var_4 == 0 ? indices.field_0 : (_temp_var_4 == 1 ? indices.field_1 : (_temp_var_4 == 2 ? indices.field_2 : (_temp_var_4 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_3 == 0 ? indices.field_0 : (_temp_var_3 == 1 ? indices.field_1 : (_temp_var_3 == 2 ? indices.field_2 : (_temp_var_3 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_2 == 0 ? indices.field_0 : (_temp_var_2 == 1 ? indices.field_1 : (_temp_var_2 == 2 ? indices.field_2 : (_temp_var_2 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_5 = ((({ int _temp_var_6 = ((({ int _temp_var_7 = ((i % 4));
        (_temp_var_7 == 0 ? indices.field_0 : (_temp_var_7 == 1 ? indices.field_1 : (_temp_var_7 == 2 ? indices.field_2 : (_temp_var_7 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_6 == 0 ? indices.field_0 : (_temp_var_6 == 1 ? indices.field_1 : (_temp_var_6 == 2 ? indices.field_2 : (_temp_var_6 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_5 == 0 ? indices.field_0 : (_temp_var_5 == 1 ? indices.field_1 : (_temp_var_5 == 2 ? indices.field_2 : (_temp_var_5 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_7_ is already defined
#ifndef _block_k_7__func
#define _block_k_7__func
__device__ int _block_k_7_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_8 = ((({ int _temp_var_9 = ((({ int _temp_var_10 = ((i % 4));
        (_temp_var_10 == 0 ? indices.field_0 : (_temp_var_10 == 1 ? indices.field_1 : (_temp_var_10 == 2 ? indices.field_2 : (_temp_var_10 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_9 == 0 ? indices.field_0 : (_temp_var_9 == 1 ? indices.field_1 : (_temp_var_9 == 2 ? indices.field_2 : (_temp_var_9 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_8 == 0 ? indices.field_0 : (_temp_var_8 == 1 ? indices.field_1 : (_temp_var_8 == 2 ? indices.field_2 : (_temp_var_8 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_9_ is already defined
#ifndef _block_k_9__func
#define _block_k_9__func
__device__ int _block_k_9_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_11 = ((({ int _temp_var_12 = ((({ int _temp_var_13 = ((i % 4));
        (_temp_var_13 == 0 ? indices.field_0 : (_temp_var_13 == 1 ? indices.field_1 : (_temp_var_13 == 2 ? indices.field_2 : (_temp_var_13 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_12 == 0 ? indices.field_0 : (_temp_var_12 == 1 ? indices.field_1 : (_temp_var_12 == 2 ? indices.field_2 : (_temp_var_12 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_11 == 0 ? indices.field_0 : (_temp_var_11 == 1 ? indices.field_1 : (_temp_var_11 == 2 ? indices.field_2 : (_temp_var_11 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_14 = ((({ int _temp_var_15 = ((({ int _temp_var_16 = ((i % 4));
        (_temp_var_16 == 0 ? indices.field_0 : (_temp_var_16 == 1 ? indices.field_1 : (_temp_var_16 == 2 ? indices.field_2 : (_temp_var_16 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_15 == 0 ? indices.field_0 : (_temp_var_15 == 1 ? indices.field_1 : (_temp_var_15 == 2 ? indices.field_2 : (_temp_var_15 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_14 == 0 ? indices.field_0 : (_temp_var_14 == 1 ? indices.field_1 : (_temp_var_14 == 2 ? indices.field_2 : (_temp_var_14 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_13_ is already defined
#ifndef _block_k_13__func
#define _block_k_13__func
__device__ int _block_k_13_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_17 = ((({ int _temp_var_18 = ((({ int _temp_var_19 = ((i % 4));
        (_temp_var_19 == 0 ? indices.field_0 : (_temp_var_19 == 1 ? indices.field_1 : (_temp_var_19 == 2 ? indices.field_2 : (_temp_var_19 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_18 == 0 ? indices.field_0 : (_temp_var_18 == 1 ? indices.field_1 : (_temp_var_18 == 2 ? indices.field_2 : (_temp_var_18 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_17 == 0 ? indices.field_0 : (_temp_var_17 == 1 ? indices.field_1 : (_temp_var_17 == 2 ? indices.field_2 : (_temp_var_17 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_15_ is already defined
#ifndef _block_k_15__func
#define _block_k_15__func
__device__ int _block_k_15_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_20 = ((({ int _temp_var_21 = ((({ int _temp_var_22 = ((i % 4));
        (_temp_var_22 == 0 ? indices.field_0 : (_temp_var_22 == 1 ? indices.field_1 : (_temp_var_22 == 2 ? indices.field_2 : (_temp_var_22 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_21 == 0 ? indices.field_0 : (_temp_var_21 == 1 ? indices.field_1 : (_temp_var_21 == 2 ? indices.field_2 : (_temp_var_21 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_20 == 0 ? indices.field_0 : (_temp_var_20 == 1 ? indices.field_1 : (_temp_var_20 == 2 ? indices.field_2 : (_temp_var_20 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_23 = ((({ int _temp_var_24 = ((({ int _temp_var_25 = ((i % 4));
        (_temp_var_25 == 0 ? indices.field_0 : (_temp_var_25 == 1 ? indices.field_1 : (_temp_var_25 == 2 ? indices.field_2 : (_temp_var_25 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_24 == 0 ? indices.field_0 : (_temp_var_24 == 1 ? indices.field_1 : (_temp_var_24 == 2 ? indices.field_2 : (_temp_var_24 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_23 == 0 ? indices.field_0 : (_temp_var_23 == 1 ? indices.field_1 : (_temp_var_23 == 2 ? indices.field_2 : (_temp_var_23 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_26 = ((({ int _temp_var_27 = ((({ int _temp_var_28 = ((i % 4));
        (_temp_var_28 == 0 ? indices.field_0 : (_temp_var_28 == 1 ? indices.field_1 : (_temp_var_28 == 2 ? indices.field_2 : (_temp_var_28 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_27 == 0 ? indices.field_0 : (_temp_var_27 == 1 ? indices.field_1 : (_temp_var_27 == 2 ? indices.field_2 : (_temp_var_27 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_26 == 0 ? indices.field_0 : (_temp_var_26 == 1 ? indices.field_1 : (_temp_var_26 == 2 ? indices.field_2 : (_temp_var_26 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_21_ is already defined
#ifndef _block_k_21__func
#define _block_k_21__func
__device__ int _block_k_21_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_29 = ((({ int _temp_var_30 = ((({ int _temp_var_31 = ((i % 4));
        (_temp_var_31 == 0 ? indices.field_0 : (_temp_var_31 == 1 ? indices.field_1 : (_temp_var_31 == 2 ? indices.field_2 : (_temp_var_31 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_30 == 0 ? indices.field_0 : (_temp_var_30 == 1 ? indices.field_1 : (_temp_var_30 == 2 ? indices.field_2 : (_temp_var_30 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_29 == 0 ? indices.field_0 : (_temp_var_29 == 1 ? indices.field_1 : (_temp_var_29 == 2 ? indices.field_2 : (_temp_var_29 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_23_ is already defined
#ifndef _block_k_23__func
#define _block_k_23__func
__device__ int _block_k_23_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_32 = ((({ int _temp_var_33 = ((({ int _temp_var_34 = ((i % 4));
        (_temp_var_34 == 0 ? indices.field_0 : (_temp_var_34 == 1 ? indices.field_1 : (_temp_var_34 == 2 ? indices.field_2 : (_temp_var_34 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_33 == 0 ? indices.field_0 : (_temp_var_33 == 1 ? indices.field_1 : (_temp_var_33 == 2 ? indices.field_2 : (_temp_var_33 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_32 == 0 ? indices.field_0 : (_temp_var_32 == 1 ? indices.field_1 : (_temp_var_32 == 2 ? indices.field_2 : (_temp_var_32 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_25_ is already defined
#ifndef _block_k_25__func
#define _block_k_25__func
__device__ int _block_k_25_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_35 = ((({ int _temp_var_36 = ((({ int _temp_var_37 = ((i % 4));
        (_temp_var_37 == 0 ? indices.field_0 : (_temp_var_37 == 1 ? indices.field_1 : (_temp_var_37 == 2 ? indices.field_2 : (_temp_var_37 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_36 == 0 ? indices.field_0 : (_temp_var_36 == 1 ? indices.field_1 : (_temp_var_36 == 2 ? indices.field_2 : (_temp_var_36 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_35 == 0 ? indices.field_0 : (_temp_var_35 == 1 ? indices.field_1 : (_temp_var_35 == 2 ? indices.field_2 : (_temp_var_35 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_27_ is already defined
#ifndef _block_k_27__func
#define _block_k_27__func
__device__ int _block_k_27_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_38 = ((({ int _temp_var_39 = ((({ int _temp_var_40 = ((i % 4));
        (_temp_var_40 == 0 ? indices.field_0 : (_temp_var_40 == 1 ? indices.field_1 : (_temp_var_40 == 2 ? indices.field_2 : (_temp_var_40 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_39 == 0 ? indices.field_0 : (_temp_var_39 == 1 ? indices.field_1 : (_temp_var_39 == 2 ? indices.field_2 : (_temp_var_39 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_38 == 0 ? indices.field_0 : (_temp_var_38 == 1 ? indices.field_1 : (_temp_var_38 == 2 ? indices.field_2 : (_temp_var_38 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_29_ is already defined
#ifndef _block_k_29__func
#define _block_k_29__func
__device__ int _block_k_29_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_41 = ((({ int _temp_var_42 = ((({ int _temp_var_43 = ((i % 4));
        (_temp_var_43 == 0 ? indices.field_0 : (_temp_var_43 == 1 ? indices.field_1 : (_temp_var_43 == 2 ? indices.field_2 : (_temp_var_43 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_42 == 0 ? indices.field_0 : (_temp_var_42 == 1 ? indices.field_1 : (_temp_var_42 == 2 ? indices.field_2 : (_temp_var_42 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_41 == 0 ? indices.field_0 : (_temp_var_41 == 1 ? indices.field_1 : (_temp_var_41 == 2 ? indices.field_2 : (_temp_var_41 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_31_ is already defined
#ifndef _block_k_31__func
#define _block_k_31__func
__device__ int _block_k_31_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_44 = ((({ int _temp_var_45 = ((({ int _temp_var_46 = ((i % 4));
        (_temp_var_46 == 0 ? indices.field_0 : (_temp_var_46 == 1 ? indices.field_1 : (_temp_var_46 == 2 ? indices.field_2 : (_temp_var_46 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_45 == 0 ? indices.field_0 : (_temp_var_45 == 1 ? indices.field_1 : (_temp_var_45 == 2 ? indices.field_2 : (_temp_var_45 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_44 == 0 ? indices.field_0 : (_temp_var_44 == 1 ? indices.field_1 : (_temp_var_44 == 2 ? indices.field_2 : (_temp_var_44 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_33_ is already defined
#ifndef _block_k_33__func
#define _block_k_33__func
__device__ int _block_k_33_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_47 = ((({ int _temp_var_48 = ((({ int _temp_var_49 = ((i % 4));
        (_temp_var_49 == 0 ? indices.field_0 : (_temp_var_49 == 1 ? indices.field_1 : (_temp_var_49 == 2 ? indices.field_2 : (_temp_var_49 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_48 == 0 ? indices.field_0 : (_temp_var_48 == 1 ? indices.field_1 : (_temp_var_48 == 2 ? indices.field_2 : (_temp_var_48 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_47 == 0 ? indices.field_0 : (_temp_var_47 == 1 ? indices.field_1 : (_temp_var_47 == 2 ? indices.field_2 : (_temp_var_47 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_35_ is already defined
#ifndef _block_k_35__func
#define _block_k_35__func
__device__ int _block_k_35_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_50 = ((({ int _temp_var_51 = ((({ int _temp_var_52 = ((i % 4));
        (_temp_var_52 == 0 ? indices.field_0 : (_temp_var_52 == 1 ? indices.field_1 : (_temp_var_52 == 2 ? indices.field_2 : (_temp_var_52 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_51 == 0 ? indices.field_0 : (_temp_var_51 == 1 ? indices.field_1 : (_temp_var_51 == 2 ? indices.field_2 : (_temp_var_51 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_50 == 0 ? indices.field_0 : (_temp_var_50 == 1 ? indices.field_1 : (_temp_var_50 == 2 ? indices.field_2 : (_temp_var_50 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_53 = ((({ int _temp_var_54 = ((({ int _temp_var_55 = ((i % 4));
        (_temp_var_55 == 0 ? indices.field_0 : (_temp_var_55 == 1 ? indices.field_1 : (_temp_var_55 == 2 ? indices.field_2 : (_temp_var_55 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_54 == 0 ? indices.field_0 : (_temp_var_54 == 1 ? indices.field_1 : (_temp_var_54 == 2 ? indices.field_2 : (_temp_var_54 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_53 == 0 ? indices.field_0 : (_temp_var_53 == 1 ? indices.field_1 : (_temp_var_53 == 2 ? indices.field_2 : (_temp_var_53 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_39_ is already defined
#ifndef _block_k_39__func
#define _block_k_39__func
__device__ int _block_k_39_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_56 = ((({ int _temp_var_57 = ((({ int _temp_var_58 = ((i % 4));
        (_temp_var_58 == 0 ? indices.field_0 : (_temp_var_58 == 1 ? indices.field_1 : (_temp_var_58 == 2 ? indices.field_2 : (_temp_var_58 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_57 == 0 ? indices.field_0 : (_temp_var_57 == 1 ? indices.field_1 : (_temp_var_57 == 2 ? indices.field_2 : (_temp_var_57 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_56 == 0 ? indices.field_0 : (_temp_var_56 == 1 ? indices.field_1 : (_temp_var_56 == 2 ? indices.field_2 : (_temp_var_56 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_41_ is already defined
#ifndef _block_k_41__func
#define _block_k_41__func
__device__ int _block_k_41_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_59 = ((({ int _temp_var_60 = ((({ int _temp_var_61 = ((i % 4));
        (_temp_var_61 == 0 ? indices.field_0 : (_temp_var_61 == 1 ? indices.field_1 : (_temp_var_61 == 2 ? indices.field_2 : (_temp_var_61 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_60 == 0 ? indices.field_0 : (_temp_var_60 == 1 ? indices.field_1 : (_temp_var_60 == 2 ? indices.field_2 : (_temp_var_60 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_59 == 0 ? indices.field_0 : (_temp_var_59 == 1 ? indices.field_1 : (_temp_var_59 == 2 ? indices.field_2 : (_temp_var_59 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_43_ is already defined
#ifndef _block_k_43__func
#define _block_k_43__func
__device__ int _block_k_43_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_62 = ((({ int _temp_var_63 = ((({ int _temp_var_64 = ((i % 4));
        (_temp_var_64 == 0 ? indices.field_0 : (_temp_var_64 == 1 ? indices.field_1 : (_temp_var_64 == 2 ? indices.field_2 : (_temp_var_64 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_63 == 0 ? indices.field_0 : (_temp_var_63 == 1 ? indices.field_1 : (_temp_var_63 == 2 ? indices.field_2 : (_temp_var_63 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_62 == 0 ? indices.field_0 : (_temp_var_62 == 1 ? indices.field_1 : (_temp_var_62 == 2 ? indices.field_2 : (_temp_var_62 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_45_ is already defined
#ifndef _block_k_45__func
#define _block_k_45__func
__device__ int _block_k_45_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_65 = ((({ int _temp_var_66 = ((({ int _temp_var_67 = ((i % 4));
        (_temp_var_67 == 0 ? indices.field_0 : (_temp_var_67 == 1 ? indices.field_1 : (_temp_var_67 == 2 ? indices.field_2 : (_temp_var_67 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_66 == 0 ? indices.field_0 : (_temp_var_66 == 1 ? indices.field_1 : (_temp_var_66 == 2 ? indices.field_2 : (_temp_var_66 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_65 == 0 ? indices.field_0 : (_temp_var_65 == 1 ? indices.field_1 : (_temp_var_65 == 2 ? indices.field_2 : (_temp_var_65 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_47_ is already defined
#ifndef _block_k_47__func
#define _block_k_47__func
__device__ int _block_k_47_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_68 = ((({ int _temp_var_69 = ((({ int _temp_var_70 = ((i % 4));
        (_temp_var_70 == 0 ? indices.field_0 : (_temp_var_70 == 1 ? indices.field_1 : (_temp_var_70 == 2 ? indices.field_2 : (_temp_var_70 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_69 == 0 ? indices.field_0 : (_temp_var_69 == 1 ? indices.field_1 : (_temp_var_69 == 2 ? indices.field_2 : (_temp_var_69 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_68 == 0 ? indices.field_0 : (_temp_var_68 == 1 ? indices.field_1 : (_temp_var_68 == 2 ? indices.field_2 : (_temp_var_68 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_49_ is already defined
#ifndef _block_k_49__func
#define _block_k_49__func
__device__ int _block_k_49_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_71 = ((({ int _temp_var_72 = ((({ int _temp_var_73 = ((i % 4));
        (_temp_var_73 == 0 ? indices.field_0 : (_temp_var_73 == 1 ? indices.field_1 : (_temp_var_73 == 2 ? indices.field_2 : (_temp_var_73 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_72 == 0 ? indices.field_0 : (_temp_var_72 == 1 ? indices.field_1 : (_temp_var_72 == 2 ? indices.field_2 : (_temp_var_72 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_71 == 0 ? indices.field_0 : (_temp_var_71 == 1 ? indices.field_1 : (_temp_var_71 == 2 ? indices.field_2 : (_temp_var_71 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_51_ is already defined
#ifndef _block_k_51__func
#define _block_k_51__func
__device__ int _block_k_51_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_74 = ((({ int _temp_var_75 = ((({ int _temp_var_76 = ((i % 4));
        (_temp_var_76 == 0 ? indices.field_0 : (_temp_var_76 == 1 ? indices.field_1 : (_temp_var_76 == 2 ? indices.field_2 : (_temp_var_76 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_75 == 0 ? indices.field_0 : (_temp_var_75 == 1 ? indices.field_1 : (_temp_var_75 == 2 ? indices.field_2 : (_temp_var_75 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_74 == 0 ? indices.field_0 : (_temp_var_74 == 1 ? indices.field_1 : (_temp_var_74 == 2 ? indices.field_2 : (_temp_var_74 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_53_ is already defined
#ifndef _block_k_53__func
#define _block_k_53__func
__device__ int _block_k_53_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_77 = ((({ int _temp_var_78 = ((({ int _temp_var_79 = ((i % 4));
        (_temp_var_79 == 0 ? indices.field_0 : (_temp_var_79 == 1 ? indices.field_1 : (_temp_var_79 == 2 ? indices.field_2 : (_temp_var_79 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_78 == 0 ? indices.field_0 : (_temp_var_78 == 1 ? indices.field_1 : (_temp_var_78 == 2 ? indices.field_2 : (_temp_var_78 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_77 == 0 ? indices.field_0 : (_temp_var_77 == 1 ? indices.field_1 : (_temp_var_77 == 2 ? indices.field_2 : (_temp_var_77 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_55_ is already defined
#ifndef _block_k_55__func
#define _block_k_55__func
__device__ int _block_k_55_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_80 = ((({ int _temp_var_81 = ((({ int _temp_var_82 = ((i % 4));
        (_temp_var_82 == 0 ? indices.field_0 : (_temp_var_82 == 1 ? indices.field_1 : (_temp_var_82 == 2 ? indices.field_2 : (_temp_var_82 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_81 == 0 ? indices.field_0 : (_temp_var_81 == 1 ? indices.field_1 : (_temp_var_81 == 2 ? indices.field_2 : (_temp_var_81 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_80 == 0 ? indices.field_0 : (_temp_var_80 == 1 ? indices.field_1 : (_temp_var_80 == 2 ? indices.field_2 : (_temp_var_80 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_57_ is already defined
#ifndef _block_k_57__func
#define _block_k_57__func
__device__ int _block_k_57_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_83 = ((({ int _temp_var_84 = ((({ int _temp_var_85 = ((i % 4));
        (_temp_var_85 == 0 ? indices.field_0 : (_temp_var_85 == 1 ? indices.field_1 : (_temp_var_85 == 2 ? indices.field_2 : (_temp_var_85 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_84 == 0 ? indices.field_0 : (_temp_var_84 == 1 ? indices.field_1 : (_temp_var_84 == 2 ? indices.field_2 : (_temp_var_84 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_83 == 0 ? indices.field_0 : (_temp_var_83 == 1 ? indices.field_1 : (_temp_var_83 == 2 ? indices.field_2 : (_temp_var_83 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_59_ is already defined
#ifndef _block_k_59__func
#define _block_k_59__func
__device__ int _block_k_59_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_86 = ((({ int _temp_var_87 = ((({ int _temp_var_88 = ((i % 4));
        (_temp_var_88 == 0 ? indices.field_0 : (_temp_var_88 == 1 ? indices.field_1 : (_temp_var_88 == 2 ? indices.field_2 : (_temp_var_88 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_87 == 0 ? indices.field_0 : (_temp_var_87 == 1 ? indices.field_1 : (_temp_var_87 == 2 ? indices.field_2 : (_temp_var_87 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_86 == 0 ? indices.field_0 : (_temp_var_86 == 1 ? indices.field_1 : (_temp_var_86 == 2 ? indices.field_2 : (_temp_var_86 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_61_ is already defined
#ifndef _block_k_61__func
#define _block_k_61__func
__device__ int _block_k_61_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_89 = ((({ int _temp_var_90 = ((({ int _temp_var_91 = ((i % 4));
        (_temp_var_91 == 0 ? indices.field_0 : (_temp_var_91 == 1 ? indices.field_1 : (_temp_var_91 == 2 ? indices.field_2 : (_temp_var_91 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_90 == 0 ? indices.field_0 : (_temp_var_90 == 1 ? indices.field_1 : (_temp_var_90 == 2 ? indices.field_2 : (_temp_var_90 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_89 == 0 ? indices.field_0 : (_temp_var_89 == 1 ? indices.field_1 : (_temp_var_89 == 2 ? indices.field_2 : (_temp_var_89 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_63_ is already defined
#ifndef _block_k_63__func
#define _block_k_63__func
__device__ int _block_k_63_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_92 = ((({ int _temp_var_93 = ((({ int _temp_var_94 = ((i % 4));
        (_temp_var_94 == 0 ? indices.field_0 : (_temp_var_94 == 1 ? indices.field_1 : (_temp_var_94 == 2 ? indices.field_2 : (_temp_var_94 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_93 == 0 ? indices.field_0 : (_temp_var_93 == 1 ? indices.field_1 : (_temp_var_93 == 2 ? indices.field_2 : (_temp_var_93 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_92 == 0 ? indices.field_0 : (_temp_var_92 == 1 ? indices.field_1 : (_temp_var_92 == 2 ? indices.field_2 : (_temp_var_92 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_65_ is already defined
#ifndef _block_k_65__func
#define _block_k_65__func
__device__ int _block_k_65_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_95 = ((({ int _temp_var_96 = ((({ int _temp_var_97 = ((i % 4));
        (_temp_var_97 == 0 ? indices.field_0 : (_temp_var_97 == 1 ? indices.field_1 : (_temp_var_97 == 2 ? indices.field_2 : (_temp_var_97 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_96 == 0 ? indices.field_0 : (_temp_var_96 == 1 ? indices.field_1 : (_temp_var_96 == 2 ? indices.field_2 : (_temp_var_96 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_95 == 0 ? indices.field_0 : (_temp_var_95 == 1 ? indices.field_1 : (_temp_var_95 == 2 ? indices.field_2 : (_temp_var_95 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_67_ is already defined
#ifndef _block_k_67__func
#define _block_k_67__func
__device__ int _block_k_67_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_98 = ((({ int _temp_var_99 = ((({ int _temp_var_100 = ((i % 4));
        (_temp_var_100 == 0 ? indices.field_0 : (_temp_var_100 == 1 ? indices.field_1 : (_temp_var_100 == 2 ? indices.field_2 : (_temp_var_100 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_99 == 0 ? indices.field_0 : (_temp_var_99 == 1 ? indices.field_1 : (_temp_var_99 == 2 ? indices.field_2 : (_temp_var_99 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_98 == 0 ? indices.field_0 : (_temp_var_98 == 1 ? indices.field_1 : (_temp_var_98 == 2 ? indices.field_2 : (_temp_var_98 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_69_ is already defined
#ifndef _block_k_69__func
#define _block_k_69__func
__device__ int _block_k_69_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_101 = ((({ int _temp_var_102 = ((({ int _temp_var_103 = ((i % 4));
        (_temp_var_103 == 0 ? indices.field_0 : (_temp_var_103 == 1 ? indices.field_1 : (_temp_var_103 == 2 ? indices.field_2 : (_temp_var_103 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_102 == 0 ? indices.field_0 : (_temp_var_102 == 1 ? indices.field_1 : (_temp_var_102 == 2 ? indices.field_2 : (_temp_var_102 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_101 == 0 ? indices.field_0 : (_temp_var_101 == 1 ? indices.field_1 : (_temp_var_101 == 2 ? indices.field_2 : (_temp_var_101 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_71_ is already defined
#ifndef _block_k_71__func
#define _block_k_71__func
__device__ int _block_k_71_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_104 = ((({ int _temp_var_105 = ((({ int _temp_var_106 = ((i % 4));
        (_temp_var_106 == 0 ? indices.field_0 : (_temp_var_106 == 1 ? indices.field_1 : (_temp_var_106 == 2 ? indices.field_2 : (_temp_var_106 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_105 == 0 ? indices.field_0 : (_temp_var_105 == 1 ? indices.field_1 : (_temp_var_105 == 2 ? indices.field_2 : (_temp_var_105 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_104 == 0 ? indices.field_0 : (_temp_var_104 == 1 ? indices.field_1 : (_temp_var_104 == 2 ? indices.field_2 : (_temp_var_104 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_73_ is already defined
#ifndef _block_k_73__func
#define _block_k_73__func
__device__ int _block_k_73_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_107 = ((({ int _temp_var_108 = ((({ int _temp_var_109 = ((i % 4));
        (_temp_var_109 == 0 ? indices.field_0 : (_temp_var_109 == 1 ? indices.field_1 : (_temp_var_109 == 2 ? indices.field_2 : (_temp_var_109 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_108 == 0 ? indices.field_0 : (_temp_var_108 == 1 ? indices.field_1 : (_temp_var_108 == 2 ? indices.field_2 : (_temp_var_108 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_107 == 0 ? indices.field_0 : (_temp_var_107 == 1 ? indices.field_1 : (_temp_var_107 == 2 ? indices.field_2 : (_temp_var_107 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_75_ is already defined
#ifndef _block_k_75__func
#define _block_k_75__func
__device__ int _block_k_75_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_110 = ((({ int _temp_var_111 = ((({ int _temp_var_112 = ((i % 4));
        (_temp_var_112 == 0 ? indices.field_0 : (_temp_var_112 == 1 ? indices.field_1 : (_temp_var_112 == 2 ? indices.field_2 : (_temp_var_112 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_111 == 0 ? indices.field_0 : (_temp_var_111 == 1 ? indices.field_1 : (_temp_var_111 == 2 ? indices.field_2 : (_temp_var_111 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_110 == 0 ? indices.field_0 : (_temp_var_110 == 1 ? indices.field_1 : (_temp_var_110 == 2 ? indices.field_2 : (_temp_var_110 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_77_ is already defined
#ifndef _block_k_77__func
#define _block_k_77__func
__device__ int _block_k_77_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_113 = ((({ int _temp_var_114 = ((({ int _temp_var_115 = ((i % 4));
        (_temp_var_115 == 0 ? indices.field_0 : (_temp_var_115 == 1 ? indices.field_1 : (_temp_var_115 == 2 ? indices.field_2 : (_temp_var_115 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_114 == 0 ? indices.field_0 : (_temp_var_114 == 1 ? indices.field_1 : (_temp_var_114 == 2 ? indices.field_2 : (_temp_var_114 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_113 == 0 ? indices.field_0 : (_temp_var_113 == 1 ? indices.field_1 : (_temp_var_113 == 2 ? indices.field_2 : (_temp_var_113 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_79_ is already defined
#ifndef _block_k_79__func
#define _block_k_79__func
__device__ int _block_k_79_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_116 = ((({ int _temp_var_117 = ((({ int _temp_var_118 = ((i % 4));
        (_temp_var_118 == 0 ? indices.field_0 : (_temp_var_118 == 1 ? indices.field_1 : (_temp_var_118 == 2 ? indices.field_2 : (_temp_var_118 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_117 == 0 ? indices.field_0 : (_temp_var_117 == 1 ? indices.field_1 : (_temp_var_117 == 2 ? indices.field_2 : (_temp_var_117 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_116 == 0 ? indices.field_0 : (_temp_var_116 == 1 ? indices.field_1 : (_temp_var_116 == 2 ? indices.field_2 : (_temp_var_116 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_81_ is already defined
#ifndef _block_k_81__func
#define _block_k_81__func
__device__ int _block_k_81_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_119 = ((({ int _temp_var_120 = ((({ int _temp_var_121 = ((i % 4));
        (_temp_var_121 == 0 ? indices.field_0 : (_temp_var_121 == 1 ? indices.field_1 : (_temp_var_121 == 2 ? indices.field_2 : (_temp_var_121 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_120 == 0 ? indices.field_0 : (_temp_var_120 == 1 ? indices.field_1 : (_temp_var_120 == 2 ? indices.field_2 : (_temp_var_120 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_119 == 0 ? indices.field_0 : (_temp_var_119 == 1 ? indices.field_1 : (_temp_var_119 == 2 ? indices.field_2 : (_temp_var_119 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_83_ is already defined
#ifndef _block_k_83__func
#define _block_k_83__func
__device__ int _block_k_83_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_122 = ((({ int _temp_var_123 = ((({ int _temp_var_124 = ((i % 4));
        (_temp_var_124 == 0 ? indices.field_0 : (_temp_var_124 == 1 ? indices.field_1 : (_temp_var_124 == 2 ? indices.field_2 : (_temp_var_124 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_123 == 0 ? indices.field_0 : (_temp_var_123 == 1 ? indices.field_1 : (_temp_var_123 == 2 ? indices.field_2 : (_temp_var_123 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_122 == 0 ? indices.field_0 : (_temp_var_122 == 1 ? indices.field_1 : (_temp_var_122 == 2 ? indices.field_2 : (_temp_var_122 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_85_ is already defined
#ifndef _block_k_85__func
#define _block_k_85__func
__device__ int _block_k_85_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_125 = ((({ int _temp_var_126 = ((({ int _temp_var_127 = ((i % 4));
        (_temp_var_127 == 0 ? indices.field_0 : (_temp_var_127 == 1 ? indices.field_1 : (_temp_var_127 == 2 ? indices.field_2 : (_temp_var_127 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_126 == 0 ? indices.field_0 : (_temp_var_126 == 1 ? indices.field_1 : (_temp_var_126 == 2 ? indices.field_2 : (_temp_var_126 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_125 == 0 ? indices.field_0 : (_temp_var_125 == 1 ? indices.field_1 : (_temp_var_125 == 2 ? indices.field_2 : (_temp_var_125 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_87_ is already defined
#ifndef _block_k_87__func
#define _block_k_87__func
__device__ int _block_k_87_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_128 = ((({ int _temp_var_129 = ((({ int _temp_var_130 = ((i % 4));
        (_temp_var_130 == 0 ? indices.field_0 : (_temp_var_130 == 1 ? indices.field_1 : (_temp_var_130 == 2 ? indices.field_2 : (_temp_var_130 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_129 == 0 ? indices.field_0 : (_temp_var_129 == 1 ? indices.field_1 : (_temp_var_129 == 2 ? indices.field_2 : (_temp_var_129 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_128 == 0 ? indices.field_0 : (_temp_var_128 == 1 ? indices.field_1 : (_temp_var_128 == 2 ? indices.field_2 : (_temp_var_128 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_131 = ((({ int _temp_var_132 = ((({ int _temp_var_133 = ((i % 4));
        (_temp_var_133 == 0 ? indices.field_0 : (_temp_var_133 == 1 ? indices.field_1 : (_temp_var_133 == 2 ? indices.field_2 : (_temp_var_133 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_132 == 0 ? indices.field_0 : (_temp_var_132 == 1 ? indices.field_1 : (_temp_var_132 == 2 ? indices.field_2 : (_temp_var_132 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_131 == 0 ? indices.field_0 : (_temp_var_131 == 1 ? indices.field_1 : (_temp_var_131 == 2 ? indices.field_2 : (_temp_var_131 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_91_ is already defined
#ifndef _block_k_91__func
#define _block_k_91__func
__device__ int _block_k_91_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_134 = ((({ int _temp_var_135 = ((({ int _temp_var_136 = ((i % 4));
        (_temp_var_136 == 0 ? indices.field_0 : (_temp_var_136 == 1 ? indices.field_1 : (_temp_var_136 == 2 ? indices.field_2 : (_temp_var_136 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_135 == 0 ? indices.field_0 : (_temp_var_135 == 1 ? indices.field_1 : (_temp_var_135 == 2 ? indices.field_2 : (_temp_var_135 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_134 == 0 ? indices.field_0 : (_temp_var_134 == 1 ? indices.field_1 : (_temp_var_134 == 2 ? indices.field_2 : (_temp_var_134 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_137 = ((({ int _temp_var_138 = ((({ int _temp_var_139 = ((i % 4));
        (_temp_var_139 == 0 ? indices.field_0 : (_temp_var_139 == 1 ? indices.field_1 : (_temp_var_139 == 2 ? indices.field_2 : (_temp_var_139 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_138 == 0 ? indices.field_0 : (_temp_var_138 == 1 ? indices.field_1 : (_temp_var_138 == 2 ? indices.field_2 : (_temp_var_138 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_137 == 0 ? indices.field_0 : (_temp_var_137 == 1 ? indices.field_1 : (_temp_var_137 == 2 ? indices.field_2 : (_temp_var_137 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_95_ is already defined
#ifndef _block_k_95__func
#define _block_k_95__func
__device__ int _block_k_95_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_140 = ((({ int _temp_var_141 = ((({ int _temp_var_142 = ((i % 4));
        (_temp_var_142 == 0 ? indices.field_0 : (_temp_var_142 == 1 ? indices.field_1 : (_temp_var_142 == 2 ? indices.field_2 : (_temp_var_142 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_141 == 0 ? indices.field_0 : (_temp_var_141 == 1 ? indices.field_1 : (_temp_var_141 == 2 ? indices.field_2 : (_temp_var_141 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_140 == 0 ? indices.field_0 : (_temp_var_140 == 1 ? indices.field_1 : (_temp_var_140 == 2 ? indices.field_2 : (_temp_var_140 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_97_ is already defined
#ifndef _block_k_97__func
#define _block_k_97__func
__device__ int _block_k_97_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_143 = ((({ int _temp_var_144 = ((({ int _temp_var_145 = ((i % 4));
        (_temp_var_145 == 0 ? indices.field_0 : (_temp_var_145 == 1 ? indices.field_1 : (_temp_var_145 == 2 ? indices.field_2 : (_temp_var_145 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_144 == 0 ? indices.field_0 : (_temp_var_144 == 1 ? indices.field_1 : (_temp_var_144 == 2 ? indices.field_2 : (_temp_var_144 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_143 == 0 ? indices.field_0 : (_temp_var_143 == 1 ? indices.field_1 : (_temp_var_143 == 2 ? indices.field_2 : (_temp_var_143 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_99_ is already defined
#ifndef _block_k_99__func
#define _block_k_99__func
__device__ int _block_k_99_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_146 = ((({ int _temp_var_147 = ((({ int _temp_var_148 = ((i % 4));
        (_temp_var_148 == 0 ? indices.field_0 : (_temp_var_148 == 1 ? indices.field_1 : (_temp_var_148 == 2 ? indices.field_2 : (_temp_var_148 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_147 == 0 ? indices.field_0 : (_temp_var_147 == 1 ? indices.field_1 : (_temp_var_147 == 2 ? indices.field_2 : (_temp_var_147 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_146 == 0 ? indices.field_0 : (_temp_var_146 == 1 ? indices.field_1 : (_temp_var_146 == 2 ? indices.field_2 : (_temp_var_146 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_101_ is already defined
#ifndef _block_k_101__func
#define _block_k_101__func
__device__ int _block_k_101_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_149 = ((({ int _temp_var_150 = ((({ int _temp_var_151 = ((i % 4));
        (_temp_var_151 == 0 ? indices.field_0 : (_temp_var_151 == 1 ? indices.field_1 : (_temp_var_151 == 2 ? indices.field_2 : (_temp_var_151 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_150 == 0 ? indices.field_0 : (_temp_var_150 == 1 ? indices.field_1 : (_temp_var_150 == 2 ? indices.field_2 : (_temp_var_150 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_149 == 0 ? indices.field_0 : (_temp_var_149 == 1 ? indices.field_1 : (_temp_var_149 == 2 ? indices.field_2 : (_temp_var_149 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_103_ is already defined
#ifndef _block_k_103__func
#define _block_k_103__func
__device__ int _block_k_103_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_152 = ((({ int _temp_var_153 = ((({ int _temp_var_154 = ((i % 4));
        (_temp_var_154 == 0 ? indices.field_0 : (_temp_var_154 == 1 ? indices.field_1 : (_temp_var_154 == 2 ? indices.field_2 : (_temp_var_154 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_153 == 0 ? indices.field_0 : (_temp_var_153 == 1 ? indices.field_1 : (_temp_var_153 == 2 ? indices.field_2 : (_temp_var_153 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_152 == 0 ? indices.field_0 : (_temp_var_152 == 1 ? indices.field_1 : (_temp_var_152 == 2 ? indices.field_2 : (_temp_var_152 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_105_ is already defined
#ifndef _block_k_105__func
#define _block_k_105__func
__device__ int _block_k_105_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_155 = ((({ int _temp_var_156 = ((({ int _temp_var_157 = ((i % 4));
        (_temp_var_157 == 0 ? indices.field_0 : (_temp_var_157 == 1 ? indices.field_1 : (_temp_var_157 == 2 ? indices.field_2 : (_temp_var_157 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_156 == 0 ? indices.field_0 : (_temp_var_156 == 1 ? indices.field_1 : (_temp_var_156 == 2 ? indices.field_2 : (_temp_var_156 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_155 == 0 ? indices.field_0 : (_temp_var_155 == 1 ? indices.field_1 : (_temp_var_155 == 2 ? indices.field_2 : (_temp_var_155 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_107_ is already defined
#ifndef _block_k_107__func
#define _block_k_107__func
__device__ int _block_k_107_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_158 = ((({ int _temp_var_159 = ((({ int _temp_var_160 = ((i % 4));
        (_temp_var_160 == 0 ? indices.field_0 : (_temp_var_160 == 1 ? indices.field_1 : (_temp_var_160 == 2 ? indices.field_2 : (_temp_var_160 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_159 == 0 ? indices.field_0 : (_temp_var_159 == 1 ? indices.field_1 : (_temp_var_159 == 2 ? indices.field_2 : (_temp_var_159 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_158 == 0 ? indices.field_0 : (_temp_var_158 == 1 ? indices.field_1 : (_temp_var_158 == 2 ? indices.field_2 : (_temp_var_158 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_109_ is already defined
#ifndef _block_k_109__func
#define _block_k_109__func
__device__ int _block_k_109_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_161 = ((({ int _temp_var_162 = ((({ int _temp_var_163 = ((i % 4));
        (_temp_var_163 == 0 ? indices.field_0 : (_temp_var_163 == 1 ? indices.field_1 : (_temp_var_163 == 2 ? indices.field_2 : (_temp_var_163 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_162 == 0 ? indices.field_0 : (_temp_var_162 == 1 ? indices.field_1 : (_temp_var_162 == 2 ? indices.field_2 : (_temp_var_162 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_161 == 0 ? indices.field_0 : (_temp_var_161 == 1 ? indices.field_1 : (_temp_var_161 == 2 ? indices.field_2 : (_temp_var_161 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_111_ is already defined
#ifndef _block_k_111__func
#define _block_k_111__func
__device__ int _block_k_111_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_164 = ((({ int _temp_var_165 = ((({ int _temp_var_166 = ((i % 4));
        (_temp_var_166 == 0 ? indices.field_0 : (_temp_var_166 == 1 ? indices.field_1 : (_temp_var_166 == 2 ? indices.field_2 : (_temp_var_166 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_165 == 0 ? indices.field_0 : (_temp_var_165 == 1 ? indices.field_1 : (_temp_var_165 == 2 ? indices.field_2 : (_temp_var_165 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_164 == 0 ? indices.field_0 : (_temp_var_164 == 1 ? indices.field_1 : (_temp_var_164 == 2 ? indices.field_2 : (_temp_var_164 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_113_ is already defined
#ifndef _block_k_113__func
#define _block_k_113__func
__device__ int _block_k_113_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_167 = ((({ int _temp_var_168 = ((({ int _temp_var_169 = ((i % 4));
        (_temp_var_169 == 0 ? indices.field_0 : (_temp_var_169 == 1 ? indices.field_1 : (_temp_var_169 == 2 ? indices.field_2 : (_temp_var_169 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_168 == 0 ? indices.field_0 : (_temp_var_168 == 1 ? indices.field_1 : (_temp_var_168 == 2 ? indices.field_2 : (_temp_var_168 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_167 == 0 ? indices.field_0 : (_temp_var_167 == 1 ? indices.field_1 : (_temp_var_167 == 2 ? indices.field_2 : (_temp_var_167 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_115_ is already defined
#ifndef _block_k_115__func
#define _block_k_115__func
__device__ int _block_k_115_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_170 = ((({ int _temp_var_171 = ((({ int _temp_var_172 = ((i % 4));
        (_temp_var_172 == 0 ? indices.field_0 : (_temp_var_172 == 1 ? indices.field_1 : (_temp_var_172 == 2 ? indices.field_2 : (_temp_var_172 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_171 == 0 ? indices.field_0 : (_temp_var_171 == 1 ? indices.field_1 : (_temp_var_171 == 2 ? indices.field_2 : (_temp_var_171 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_170 == 0 ? indices.field_0 : (_temp_var_170 == 1 ? indices.field_1 : (_temp_var_170 == 2 ? indices.field_2 : (_temp_var_170 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_117_ is already defined
#ifndef _block_k_117__func
#define _block_k_117__func
__device__ int _block_k_117_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_173 = ((({ int _temp_var_174 = ((({ int _temp_var_175 = ((i % 4));
        (_temp_var_175 == 0 ? indices.field_0 : (_temp_var_175 == 1 ? indices.field_1 : (_temp_var_175 == 2 ? indices.field_2 : (_temp_var_175 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_174 == 0 ? indices.field_0 : (_temp_var_174 == 1 ? indices.field_1 : (_temp_var_174 == 2 ? indices.field_2 : (_temp_var_174 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_173 == 0 ? indices.field_0 : (_temp_var_173 == 1 ? indices.field_1 : (_temp_var_173 == 2 ? indices.field_2 : (_temp_var_173 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_119_ is already defined
#ifndef _block_k_119__func
#define _block_k_119__func
__device__ int _block_k_119_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_176 = ((({ int _temp_var_177 = ((({ int _temp_var_178 = ((i % 4));
        (_temp_var_178 == 0 ? indices.field_0 : (_temp_var_178 == 1 ? indices.field_1 : (_temp_var_178 == 2 ? indices.field_2 : (_temp_var_178 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_177 == 0 ? indices.field_0 : (_temp_var_177 == 1 ? indices.field_1 : (_temp_var_177 == 2 ? indices.field_2 : (_temp_var_177 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_176 == 0 ? indices.field_0 : (_temp_var_176 == 1 ? indices.field_1 : (_temp_var_176 == 2 ? indices.field_2 : (_temp_var_176 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_121_ is already defined
#ifndef _block_k_121__func
#define _block_k_121__func
__device__ int _block_k_121_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_179 = ((({ int _temp_var_180 = ((({ int _temp_var_181 = ((i % 4));
        (_temp_var_181 == 0 ? indices.field_0 : (_temp_var_181 == 1 ? indices.field_1 : (_temp_var_181 == 2 ? indices.field_2 : (_temp_var_181 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_180 == 0 ? indices.field_0 : (_temp_var_180 == 1 ? indices.field_1 : (_temp_var_180 == 2 ? indices.field_2 : (_temp_var_180 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_179 == 0 ? indices.field_0 : (_temp_var_179 == 1 ? indices.field_1 : (_temp_var_179 == 2 ? indices.field_2 : (_temp_var_179 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_123_ is already defined
#ifndef _block_k_123__func
#define _block_k_123__func
__device__ int _block_k_123_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_182 = ((({ int _temp_var_183 = ((({ int _temp_var_184 = ((i % 4));
        (_temp_var_184 == 0 ? indices.field_0 : (_temp_var_184 == 1 ? indices.field_1 : (_temp_var_184 == 2 ? indices.field_2 : (_temp_var_184 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_183 == 0 ? indices.field_0 : (_temp_var_183 == 1 ? indices.field_1 : (_temp_var_183 == 2 ? indices.field_2 : (_temp_var_183 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_182 == 0 ? indices.field_0 : (_temp_var_182 == 1 ? indices.field_1 : (_temp_var_182 == 2 ? indices.field_2 : (_temp_var_182 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_125_ is already defined
#ifndef _block_k_125__func
#define _block_k_125__func
__device__ int _block_k_125_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_185 = ((({ int _temp_var_186 = ((({ int _temp_var_187 = ((i % 4));
        (_temp_var_187 == 0 ? indices.field_0 : (_temp_var_187 == 1 ? indices.field_1 : (_temp_var_187 == 2 ? indices.field_2 : (_temp_var_187 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_186 == 0 ? indices.field_0 : (_temp_var_186 == 1 ? indices.field_1 : (_temp_var_186 == 2 ? indices.field_2 : (_temp_var_186 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_185 == 0 ? indices.field_0 : (_temp_var_185 == 1 ? indices.field_1 : (_temp_var_185 == 2 ? indices.field_2 : (_temp_var_185 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_127_ is already defined
#ifndef _block_k_127__func
#define _block_k_127__func
__device__ int _block_k_127_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_188 = ((({ int _temp_var_189 = ((({ int _temp_var_190 = ((i % 4));
        (_temp_var_190 == 0 ? indices.field_0 : (_temp_var_190 == 1 ? indices.field_1 : (_temp_var_190 == 2 ? indices.field_2 : (_temp_var_190 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_189 == 0 ? indices.field_0 : (_temp_var_189 == 1 ? indices.field_1 : (_temp_var_189 == 2 ? indices.field_2 : (_temp_var_189 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_188 == 0 ? indices.field_0 : (_temp_var_188 == 1 ? indices.field_1 : (_temp_var_188 == 2 ? indices.field_2 : (_temp_var_188 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_129_ is already defined
#ifndef _block_k_129__func
#define _block_k_129__func
__device__ int _block_k_129_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_191 = ((({ int _temp_var_192 = ((({ int _temp_var_193 = ((i % 4));
        (_temp_var_193 == 0 ? indices.field_0 : (_temp_var_193 == 1 ? indices.field_1 : (_temp_var_193 == 2 ? indices.field_2 : (_temp_var_193 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_192 == 0 ? indices.field_0 : (_temp_var_192 == 1 ? indices.field_1 : (_temp_var_192 == 2 ? indices.field_2 : (_temp_var_192 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_191 == 0 ? indices.field_0 : (_temp_var_191 == 1 ? indices.field_1 : (_temp_var_191 == 2 ? indices.field_2 : (_temp_var_191 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_131_ is already defined
#ifndef _block_k_131__func
#define _block_k_131__func
__device__ int _block_k_131_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_194 = ((({ int _temp_var_195 = ((({ int _temp_var_196 = ((i % 4));
        (_temp_var_196 == 0 ? indices.field_0 : (_temp_var_196 == 1 ? indices.field_1 : (_temp_var_196 == 2 ? indices.field_2 : (_temp_var_196 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_195 == 0 ? indices.field_0 : (_temp_var_195 == 1 ? indices.field_1 : (_temp_var_195 == 2 ? indices.field_2 : (_temp_var_195 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_194 == 0 ? indices.field_0 : (_temp_var_194 == 1 ? indices.field_1 : (_temp_var_194 == 2 ? indices.field_2 : (_temp_var_194 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_133_ is already defined
#ifndef _block_k_133__func
#define _block_k_133__func
__device__ int _block_k_133_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_197 = ((({ int _temp_var_198 = ((({ int _temp_var_199 = ((i % 4));
        (_temp_var_199 == 0 ? indices.field_0 : (_temp_var_199 == 1 ? indices.field_1 : (_temp_var_199 == 2 ? indices.field_2 : (_temp_var_199 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_198 == 0 ? indices.field_0 : (_temp_var_198 == 1 ? indices.field_1 : (_temp_var_198 == 2 ? indices.field_2 : (_temp_var_198 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_197 == 0 ? indices.field_0 : (_temp_var_197 == 1 ? indices.field_1 : (_temp_var_197 == 2 ? indices.field_2 : (_temp_var_197 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_135_ is already defined
#ifndef _block_k_135__func
#define _block_k_135__func
__device__ int _block_k_135_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_200 = ((({ int _temp_var_201 = ((({ int _temp_var_202 = ((i % 4));
        (_temp_var_202 == 0 ? indices.field_0 : (_temp_var_202 == 1 ? indices.field_1 : (_temp_var_202 == 2 ? indices.field_2 : (_temp_var_202 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_201 == 0 ? indices.field_0 : (_temp_var_201 == 1 ? indices.field_1 : (_temp_var_201 == 2 ? indices.field_2 : (_temp_var_201 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_200 == 0 ? indices.field_0 : (_temp_var_200 == 1 ? indices.field_1 : (_temp_var_200 == 2 ? indices.field_2 : (_temp_var_200 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_137_ is already defined
#ifndef _block_k_137__func
#define _block_k_137__func
__device__ int _block_k_137_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_203 = ((({ int _temp_var_204 = ((({ int _temp_var_205 = ((i % 4));
        (_temp_var_205 == 0 ? indices.field_0 : (_temp_var_205 == 1 ? indices.field_1 : (_temp_var_205 == 2 ? indices.field_2 : (_temp_var_205 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_204 == 0 ? indices.field_0 : (_temp_var_204 == 1 ? indices.field_1 : (_temp_var_204 == 2 ? indices.field_2 : (_temp_var_204 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_203 == 0 ? indices.field_0 : (_temp_var_203 == 1 ? indices.field_1 : (_temp_var_203 == 2 ? indices.field_2 : (_temp_var_203 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_139_ is already defined
#ifndef _block_k_139__func
#define _block_k_139__func
__device__ int _block_k_139_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_206 = ((({ int _temp_var_207 = ((({ int _temp_var_208 = ((i % 4));
        (_temp_var_208 == 0 ? indices.field_0 : (_temp_var_208 == 1 ? indices.field_1 : (_temp_var_208 == 2 ? indices.field_2 : (_temp_var_208 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_207 == 0 ? indices.field_0 : (_temp_var_207 == 1 ? indices.field_1 : (_temp_var_207 == 2 ? indices.field_2 : (_temp_var_207 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_206 == 0 ? indices.field_0 : (_temp_var_206 == 1 ? indices.field_1 : (_temp_var_206 == 2 ? indices.field_2 : (_temp_var_206 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_141_ is already defined
#ifndef _block_k_141__func
#define _block_k_141__func
__device__ int _block_k_141_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_209 = ((({ int _temp_var_210 = ((({ int _temp_var_211 = ((i % 4));
        (_temp_var_211 == 0 ? indices.field_0 : (_temp_var_211 == 1 ? indices.field_1 : (_temp_var_211 == 2 ? indices.field_2 : (_temp_var_211 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_210 == 0 ? indices.field_0 : (_temp_var_210 == 1 ? indices.field_1 : (_temp_var_210 == 2 ? indices.field_2 : (_temp_var_210 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_209 == 0 ? indices.field_0 : (_temp_var_209 == 1 ? indices.field_1 : (_temp_var_209 == 2 ? indices.field_2 : (_temp_var_209 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_143_ is already defined
#ifndef _block_k_143__func
#define _block_k_143__func
__device__ int _block_k_143_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_212 = ((({ int _temp_var_213 = ((({ int _temp_var_214 = ((i % 4));
        (_temp_var_214 == 0 ? indices.field_0 : (_temp_var_214 == 1 ? indices.field_1 : (_temp_var_214 == 2 ? indices.field_2 : (_temp_var_214 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_213 == 0 ? indices.field_0 : (_temp_var_213 == 1 ? indices.field_1 : (_temp_var_213 == 2 ? indices.field_2 : (_temp_var_213 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_212 == 0 ? indices.field_0 : (_temp_var_212 == 1 ? indices.field_1 : (_temp_var_212 == 2 ? indices.field_2 : (_temp_var_212 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_145_ is already defined
#ifndef _block_k_145__func
#define _block_k_145__func
__device__ int _block_k_145_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_215 = ((({ int _temp_var_216 = ((({ int _temp_var_217 = ((i % 4));
        (_temp_var_217 == 0 ? indices.field_0 : (_temp_var_217 == 1 ? indices.field_1 : (_temp_var_217 == 2 ? indices.field_2 : (_temp_var_217 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_216 == 0 ? indices.field_0 : (_temp_var_216 == 1 ? indices.field_1 : (_temp_var_216 == 2 ? indices.field_2 : (_temp_var_216 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_215 == 0 ? indices.field_0 : (_temp_var_215 == 1 ? indices.field_1 : (_temp_var_215 == 2 ? indices.field_2 : (_temp_var_215 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_147_ is already defined
#ifndef _block_k_147__func
#define _block_k_147__func
__device__ int _block_k_147_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_218 = ((({ int _temp_var_219 = ((({ int _temp_var_220 = ((i % 4));
        (_temp_var_220 == 0 ? indices.field_0 : (_temp_var_220 == 1 ? indices.field_1 : (_temp_var_220 == 2 ? indices.field_2 : (_temp_var_220 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_219 == 0 ? indices.field_0 : (_temp_var_219 == 1 ? indices.field_1 : (_temp_var_219 == 2 ? indices.field_2 : (_temp_var_219 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_218 == 0 ? indices.field_0 : (_temp_var_218 == 1 ? indices.field_1 : (_temp_var_218 == 2 ? indices.field_2 : (_temp_var_218 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_149_ is already defined
#ifndef _block_k_149__func
#define _block_k_149__func
__device__ int _block_k_149_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_221 = ((({ int _temp_var_222 = ((({ int _temp_var_223 = ((i % 4));
        (_temp_var_223 == 0 ? indices.field_0 : (_temp_var_223 == 1 ? indices.field_1 : (_temp_var_223 == 2 ? indices.field_2 : (_temp_var_223 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_222 == 0 ? indices.field_0 : (_temp_var_222 == 1 ? indices.field_1 : (_temp_var_222 == 2 ? indices.field_2 : (_temp_var_222 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_221 == 0 ? indices.field_0 : (_temp_var_221 == 1 ? indices.field_1 : (_temp_var_221 == 2 ? indices.field_2 : (_temp_var_221 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_151_ is already defined
#ifndef _block_k_151__func
#define _block_k_151__func
__device__ int _block_k_151_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_224 = ((({ int _temp_var_225 = ((({ int _temp_var_226 = ((i % 4));
        (_temp_var_226 == 0 ? indices.field_0 : (_temp_var_226 == 1 ? indices.field_1 : (_temp_var_226 == 2 ? indices.field_2 : (_temp_var_226 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_225 == 0 ? indices.field_0 : (_temp_var_225 == 1 ? indices.field_1 : (_temp_var_225 == 2 ? indices.field_2 : (_temp_var_225 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_224 == 0 ? indices.field_0 : (_temp_var_224 == 1 ? indices.field_1 : (_temp_var_224 == 2 ? indices.field_2 : (_temp_var_224 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_153_ is already defined
#ifndef _block_k_153__func
#define _block_k_153__func
__device__ int _block_k_153_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_227 = ((({ int _temp_var_228 = ((({ int _temp_var_229 = ((i % 4));
        (_temp_var_229 == 0 ? indices.field_0 : (_temp_var_229 == 1 ? indices.field_1 : (_temp_var_229 == 2 ? indices.field_2 : (_temp_var_229 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_228 == 0 ? indices.field_0 : (_temp_var_228 == 1 ? indices.field_1 : (_temp_var_228 == 2 ? indices.field_2 : (_temp_var_228 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_227 == 0 ? indices.field_0 : (_temp_var_227 == 1 ? indices.field_1 : (_temp_var_227 == 2 ? indices.field_2 : (_temp_var_227 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_155_ is already defined
#ifndef _block_k_155__func
#define _block_k_155__func
__device__ int _block_k_155_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_230 = ((({ int _temp_var_231 = ((({ int _temp_var_232 = ((i % 4));
        (_temp_var_232 == 0 ? indices.field_0 : (_temp_var_232 == 1 ? indices.field_1 : (_temp_var_232 == 2 ? indices.field_2 : (_temp_var_232 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_231 == 0 ? indices.field_0 : (_temp_var_231 == 1 ? indices.field_1 : (_temp_var_231 == 2 ? indices.field_2 : (_temp_var_231 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_230 == 0 ? indices.field_0 : (_temp_var_230 == 1 ? indices.field_1 : (_temp_var_230 == 2 ? indices.field_2 : (_temp_var_230 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_157_ is already defined
#ifndef _block_k_157__func
#define _block_k_157__func
__device__ int _block_k_157_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_233 = ((({ int _temp_var_234 = ((({ int _temp_var_235 = ((i % 4));
        (_temp_var_235 == 0 ? indices.field_0 : (_temp_var_235 == 1 ? indices.field_1 : (_temp_var_235 == 2 ? indices.field_2 : (_temp_var_235 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_234 == 0 ? indices.field_0 : (_temp_var_234 == 1 ? indices.field_1 : (_temp_var_234 == 2 ? indices.field_2 : (_temp_var_234 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_233 == 0 ? indices.field_0 : (_temp_var_233 == 1 ? indices.field_1 : (_temp_var_233 == 2 ? indices.field_2 : (_temp_var_233 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_159_ is already defined
#ifndef _block_k_159__func
#define _block_k_159__func
__device__ int _block_k_159_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_236 = ((({ int _temp_var_237 = ((({ int _temp_var_238 = ((i % 4));
        (_temp_var_238 == 0 ? indices.field_0 : (_temp_var_238 == 1 ? indices.field_1 : (_temp_var_238 == 2 ? indices.field_2 : (_temp_var_238 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_237 == 0 ? indices.field_0 : (_temp_var_237 == 1 ? indices.field_1 : (_temp_var_237 == 2 ? indices.field_2 : (_temp_var_237 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_236 == 0 ? indices.field_0 : (_temp_var_236 == 1 ? indices.field_1 : (_temp_var_236 == 2 ? indices.field_2 : (_temp_var_236 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_161_ is already defined
#ifndef _block_k_161__func
#define _block_k_161__func
__device__ int _block_k_161_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_239 = ((({ int _temp_var_240 = ((({ int _temp_var_241 = ((i % 4));
        (_temp_var_241 == 0 ? indices.field_0 : (_temp_var_241 == 1 ? indices.field_1 : (_temp_var_241 == 2 ? indices.field_2 : (_temp_var_241 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_240 == 0 ? indices.field_0 : (_temp_var_240 == 1 ? indices.field_1 : (_temp_var_240 == 2 ? indices.field_2 : (_temp_var_240 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_239 == 0 ? indices.field_0 : (_temp_var_239 == 1 ? indices.field_1 : (_temp_var_239 == 2 ? indices.field_2 : (_temp_var_239 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_163_ is already defined
#ifndef _block_k_163__func
#define _block_k_163__func
__device__ int _block_k_163_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_242 = ((({ int _temp_var_243 = ((({ int _temp_var_244 = ((i % 4));
        (_temp_var_244 == 0 ? indices.field_0 : (_temp_var_244 == 1 ? indices.field_1 : (_temp_var_244 == 2 ? indices.field_2 : (_temp_var_244 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_243 == 0 ? indices.field_0 : (_temp_var_243 == 1 ? indices.field_1 : (_temp_var_243 == 2 ? indices.field_2 : (_temp_var_243 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_242 == 0 ? indices.field_0 : (_temp_var_242 == 1 ? indices.field_1 : (_temp_var_242 == 2 ? indices.field_2 : (_temp_var_242 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_165_ is already defined
#ifndef _block_k_165__func
#define _block_k_165__func
__device__ int _block_k_165_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_245 = ((({ int _temp_var_246 = ((({ int _temp_var_247 = ((i % 4));
        (_temp_var_247 == 0 ? indices.field_0 : (_temp_var_247 == 1 ? indices.field_1 : (_temp_var_247 == 2 ? indices.field_2 : (_temp_var_247 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_246 == 0 ? indices.field_0 : (_temp_var_246 == 1 ? indices.field_1 : (_temp_var_246 == 2 ? indices.field_2 : (_temp_var_246 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_245 == 0 ? indices.field_0 : (_temp_var_245 == 1 ? indices.field_1 : (_temp_var_245 == 2 ? indices.field_2 : (_temp_var_245 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_167_ is already defined
#ifndef _block_k_167__func
#define _block_k_167__func
__device__ int _block_k_167_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_248 = ((({ int _temp_var_249 = ((({ int _temp_var_250 = ((i % 4));
        (_temp_var_250 == 0 ? indices.field_0 : (_temp_var_250 == 1 ? indices.field_1 : (_temp_var_250 == 2 ? indices.field_2 : (_temp_var_250 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_249 == 0 ? indices.field_0 : (_temp_var_249 == 1 ? indices.field_1 : (_temp_var_249 == 2 ? indices.field_2 : (_temp_var_249 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_248 == 0 ? indices.field_0 : (_temp_var_248 == 1 ? indices.field_1 : (_temp_var_248 == 2 ? indices.field_2 : (_temp_var_248 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_169_ is already defined
#ifndef _block_k_169__func
#define _block_k_169__func
__device__ int _block_k_169_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_251 = ((({ int _temp_var_252 = ((({ int _temp_var_253 = ((i % 4));
        (_temp_var_253 == 0 ? indices.field_0 : (_temp_var_253 == 1 ? indices.field_1 : (_temp_var_253 == 2 ? indices.field_2 : (_temp_var_253 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_252 == 0 ? indices.field_0 : (_temp_var_252 == 1 ? indices.field_1 : (_temp_var_252 == 2 ? indices.field_2 : (_temp_var_252 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_251 == 0 ? indices.field_0 : (_temp_var_251 == 1 ? indices.field_1 : (_temp_var_251 == 2 ? indices.field_2 : (_temp_var_251 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_171_ is already defined
#ifndef _block_k_171__func
#define _block_k_171__func
__device__ int _block_k_171_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_254 = ((({ int _temp_var_255 = ((({ int _temp_var_256 = ((i % 4));
        (_temp_var_256 == 0 ? indices.field_0 : (_temp_var_256 == 1 ? indices.field_1 : (_temp_var_256 == 2 ? indices.field_2 : (_temp_var_256 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_255 == 0 ? indices.field_0 : (_temp_var_255 == 1 ? indices.field_1 : (_temp_var_255 == 2 ? indices.field_2 : (_temp_var_255 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_254 == 0 ? indices.field_0 : (_temp_var_254 == 1 ? indices.field_1 : (_temp_var_254 == 2 ? indices.field_2 : (_temp_var_254 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_173_ is already defined
#ifndef _block_k_173__func
#define _block_k_173__func
__device__ int _block_k_173_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_257 = ((({ int _temp_var_258 = ((({ int _temp_var_259 = ((i % 4));
        (_temp_var_259 == 0 ? indices.field_0 : (_temp_var_259 == 1 ? indices.field_1 : (_temp_var_259 == 2 ? indices.field_2 : (_temp_var_259 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_258 == 0 ? indices.field_0 : (_temp_var_258 == 1 ? indices.field_1 : (_temp_var_258 == 2 ? indices.field_2 : (_temp_var_258 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_257 == 0 ? indices.field_0 : (_temp_var_257 == 1 ? indices.field_1 : (_temp_var_257 == 2 ? indices.field_2 : (_temp_var_257 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_175_ is already defined
#ifndef _block_k_175__func
#define _block_k_175__func
__device__ int _block_k_175_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_260 = ((({ int _temp_var_261 = ((({ int _temp_var_262 = ((i % 4));
        (_temp_var_262 == 0 ? indices.field_0 : (_temp_var_262 == 1 ? indices.field_1 : (_temp_var_262 == 2 ? indices.field_2 : (_temp_var_262 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_261 == 0 ? indices.field_0 : (_temp_var_261 == 1 ? indices.field_1 : (_temp_var_261 == 2 ? indices.field_2 : (_temp_var_261 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_260 == 0 ? indices.field_0 : (_temp_var_260 == 1 ? indices.field_1 : (_temp_var_260 == 2 ? indices.field_2 : (_temp_var_260 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_177_ is already defined
#ifndef _block_k_177__func
#define _block_k_177__func
__device__ int _block_k_177_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_263 = ((({ int _temp_var_264 = ((({ int _temp_var_265 = ((i % 4));
        (_temp_var_265 == 0 ? indices.field_0 : (_temp_var_265 == 1 ? indices.field_1 : (_temp_var_265 == 2 ? indices.field_2 : (_temp_var_265 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_264 == 0 ? indices.field_0 : (_temp_var_264 == 1 ? indices.field_1 : (_temp_var_264 == 2 ? indices.field_2 : (_temp_var_264 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_263 == 0 ? indices.field_0 : (_temp_var_263 == 1 ? indices.field_1 : (_temp_var_263 == 2 ? indices.field_2 : (_temp_var_263 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_179_ is already defined
#ifndef _block_k_179__func
#define _block_k_179__func
__device__ int _block_k_179_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_266 = ((({ int _temp_var_267 = ((({ int _temp_var_268 = ((i % 4));
        (_temp_var_268 == 0 ? indices.field_0 : (_temp_var_268 == 1 ? indices.field_1 : (_temp_var_268 == 2 ? indices.field_2 : (_temp_var_268 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_267 == 0 ? indices.field_0 : (_temp_var_267 == 1 ? indices.field_1 : (_temp_var_267 == 2 ? indices.field_2 : (_temp_var_267 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_266 == 0 ? indices.field_0 : (_temp_var_266 == 1 ? indices.field_1 : (_temp_var_266 == 2 ? indices.field_2 : (_temp_var_266 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_181_ is already defined
#ifndef _block_k_181__func
#define _block_k_181__func
__device__ int _block_k_181_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_269 = ((({ int _temp_var_270 = ((({ int _temp_var_271 = ((i % 4));
        (_temp_var_271 == 0 ? indices.field_0 : (_temp_var_271 == 1 ? indices.field_1 : (_temp_var_271 == 2 ? indices.field_2 : (_temp_var_271 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_270 == 0 ? indices.field_0 : (_temp_var_270 == 1 ? indices.field_1 : (_temp_var_270 == 2 ? indices.field_2 : (_temp_var_270 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_269 == 0 ? indices.field_0 : (_temp_var_269 == 1 ? indices.field_1 : (_temp_var_269 == 2 ? indices.field_2 : (_temp_var_269 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_183_ is already defined
#ifndef _block_k_183__func
#define _block_k_183__func
__device__ int _block_k_183_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_272 = ((({ int _temp_var_273 = ((({ int _temp_var_274 = ((i % 4));
        (_temp_var_274 == 0 ? indices.field_0 : (_temp_var_274 == 1 ? indices.field_1 : (_temp_var_274 == 2 ? indices.field_2 : (_temp_var_274 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_273 == 0 ? indices.field_0 : (_temp_var_273 == 1 ? indices.field_1 : (_temp_var_273 == 2 ? indices.field_2 : (_temp_var_273 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_272 == 0 ? indices.field_0 : (_temp_var_272 == 1 ? indices.field_1 : (_temp_var_272 == 2 ? indices.field_2 : (_temp_var_272 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_185_ is already defined
#ifndef _block_k_185__func
#define _block_k_185__func
__device__ int _block_k_185_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_275 = ((({ int _temp_var_276 = ((({ int _temp_var_277 = ((i % 4));
        (_temp_var_277 == 0 ? indices.field_0 : (_temp_var_277 == 1 ? indices.field_1 : (_temp_var_277 == 2 ? indices.field_2 : (_temp_var_277 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_276 == 0 ? indices.field_0 : (_temp_var_276 == 1 ? indices.field_1 : (_temp_var_276 == 2 ? indices.field_2 : (_temp_var_276 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_275 == 0 ? indices.field_0 : (_temp_var_275 == 1 ? indices.field_1 : (_temp_var_275 == 2 ? indices.field_2 : (_temp_var_275 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_187_ is already defined
#ifndef _block_k_187__func
#define _block_k_187__func
__device__ int _block_k_187_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_278 = ((({ int _temp_var_279 = ((({ int _temp_var_280 = ((i % 4));
        (_temp_var_280 == 0 ? indices.field_0 : (_temp_var_280 == 1 ? indices.field_1 : (_temp_var_280 == 2 ? indices.field_2 : (_temp_var_280 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_279 == 0 ? indices.field_0 : (_temp_var_279 == 1 ? indices.field_1 : (_temp_var_279 == 2 ? indices.field_2 : (_temp_var_279 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_278 == 0 ? indices.field_0 : (_temp_var_278 == 1 ? indices.field_1 : (_temp_var_278 == 2 ? indices.field_2 : (_temp_var_278 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_189_ is already defined
#ifndef _block_k_189__func
#define _block_k_189__func
__device__ int _block_k_189_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_281 = ((({ int _temp_var_282 = ((({ int _temp_var_283 = ((i % 4));
        (_temp_var_283 == 0 ? indices.field_0 : (_temp_var_283 == 1 ? indices.field_1 : (_temp_var_283 == 2 ? indices.field_2 : (_temp_var_283 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_282 == 0 ? indices.field_0 : (_temp_var_282 == 1 ? indices.field_1 : (_temp_var_282 == 2 ? indices.field_2 : (_temp_var_282 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_281 == 0 ? indices.field_0 : (_temp_var_281 == 1 ? indices.field_1 : (_temp_var_281 == 2 ? indices.field_2 : (_temp_var_281 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_191_ is already defined
#ifndef _block_k_191__func
#define _block_k_191__func
__device__ int _block_k_191_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_284 = ((({ int _temp_var_285 = ((({ int _temp_var_286 = ((i % 4));
        (_temp_var_286 == 0 ? indices.field_0 : (_temp_var_286 == 1 ? indices.field_1 : (_temp_var_286 == 2 ? indices.field_2 : (_temp_var_286 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_285 == 0 ? indices.field_0 : (_temp_var_285 == 1 ? indices.field_1 : (_temp_var_285 == 2 ? indices.field_2 : (_temp_var_285 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_284 == 0 ? indices.field_0 : (_temp_var_284 == 1 ? indices.field_1 : (_temp_var_284 == 2 ? indices.field_2 : (_temp_var_284 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_193_ is already defined
#ifndef _block_k_193__func
#define _block_k_193__func
__device__ int _block_k_193_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_287 = ((({ int _temp_var_288 = ((({ int _temp_var_289 = ((i % 4));
        (_temp_var_289 == 0 ? indices.field_0 : (_temp_var_289 == 1 ? indices.field_1 : (_temp_var_289 == 2 ? indices.field_2 : (_temp_var_289 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_288 == 0 ? indices.field_0 : (_temp_var_288 == 1 ? indices.field_1 : (_temp_var_288 == 2 ? indices.field_2 : (_temp_var_288 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_287 == 0 ? indices.field_0 : (_temp_var_287 == 1 ? indices.field_1 : (_temp_var_287 == 2 ? indices.field_2 : (_temp_var_287 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_195_ is already defined
#ifndef _block_k_195__func
#define _block_k_195__func
__device__ int _block_k_195_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_290 = ((({ int _temp_var_291 = ((({ int _temp_var_292 = ((i % 4));
        (_temp_var_292 == 0 ? indices.field_0 : (_temp_var_292 == 1 ? indices.field_1 : (_temp_var_292 == 2 ? indices.field_2 : (_temp_var_292 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_291 == 0 ? indices.field_0 : (_temp_var_291 == 1 ? indices.field_1 : (_temp_var_291 == 2 ? indices.field_2 : (_temp_var_291 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_290 == 0 ? indices.field_0 : (_temp_var_290 == 1 ? indices.field_1 : (_temp_var_290 == 2 ? indices.field_2 : (_temp_var_290 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_197_ is already defined
#ifndef _block_k_197__func
#define _block_k_197__func
__device__ int _block_k_197_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_293 = ((({ int _temp_var_294 = ((({ int _temp_var_295 = ((i % 4));
        (_temp_var_295 == 0 ? indices.field_0 : (_temp_var_295 == 1 ? indices.field_1 : (_temp_var_295 == 2 ? indices.field_2 : (_temp_var_295 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_294 == 0 ? indices.field_0 : (_temp_var_294 == 1 ? indices.field_1 : (_temp_var_294 == 2 ? indices.field_2 : (_temp_var_294 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_293 == 0 ? indices.field_0 : (_temp_var_293 == 1 ? indices.field_1 : (_temp_var_293 == 2 ? indices.field_2 : (_temp_var_293 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_199_ is already defined
#ifndef _block_k_199__func
#define _block_k_199__func
__device__ int _block_k_199_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_296 = ((({ int _temp_var_297 = ((({ int _temp_var_298 = ((i % 4));
        (_temp_var_298 == 0 ? indices.field_0 : (_temp_var_298 == 1 ? indices.field_1 : (_temp_var_298 == 2 ? indices.field_2 : (_temp_var_298 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_297 == 0 ? indices.field_0 : (_temp_var_297 == 1 ? indices.field_1 : (_temp_var_297 == 2 ? indices.field_2 : (_temp_var_297 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_296 == 0 ? indices.field_0 : (_temp_var_296 == 1 ? indices.field_1 : (_temp_var_296 == 2 ? indices.field_2 : (_temp_var_296 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_201_ is already defined
#ifndef _block_k_201__func
#define _block_k_201__func
__device__ int _block_k_201_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_299 = ((({ int _temp_var_300 = ((({ int _temp_var_301 = ((i % 4));
        (_temp_var_301 == 0 ? indices.field_0 : (_temp_var_301 == 1 ? indices.field_1 : (_temp_var_301 == 2 ? indices.field_2 : (_temp_var_301 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_300 == 0 ? indices.field_0 : (_temp_var_300 == 1 ? indices.field_1 : (_temp_var_300 == 2 ? indices.field_2 : (_temp_var_300 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_299 == 0 ? indices.field_0 : (_temp_var_299 == 1 ? indices.field_1 : (_temp_var_299 == 2 ? indices.field_2 : (_temp_var_299 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_203_ is already defined
#ifndef _block_k_203__func
#define _block_k_203__func
__device__ int _block_k_203_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_302 = ((({ int _temp_var_303 = ((({ int _temp_var_304 = ((i % 4));
        (_temp_var_304 == 0 ? indices.field_0 : (_temp_var_304 == 1 ? indices.field_1 : (_temp_var_304 == 2 ? indices.field_2 : (_temp_var_304 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_303 == 0 ? indices.field_0 : (_temp_var_303 == 1 ? indices.field_1 : (_temp_var_303 == 2 ? indices.field_2 : (_temp_var_303 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_302 == 0 ? indices.field_0 : (_temp_var_302 == 1 ? indices.field_1 : (_temp_var_302 == 2 ? indices.field_2 : (_temp_var_302 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_205_ is already defined
#ifndef _block_k_205__func
#define _block_k_205__func
__device__ int _block_k_205_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_305 = ((({ int _temp_var_306 = ((({ int _temp_var_307 = ((i % 4));
        (_temp_var_307 == 0 ? indices.field_0 : (_temp_var_307 == 1 ? indices.field_1 : (_temp_var_307 == 2 ? indices.field_2 : (_temp_var_307 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_306 == 0 ? indices.field_0 : (_temp_var_306 == 1 ? indices.field_1 : (_temp_var_306 == 2 ? indices.field_2 : (_temp_var_306 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_305 == 0 ? indices.field_0 : (_temp_var_305 == 1 ? indices.field_1 : (_temp_var_305 == 2 ? indices.field_2 : (_temp_var_305 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_207_ is already defined
#ifndef _block_k_207__func
#define _block_k_207__func
__device__ int _block_k_207_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_308 = ((({ int _temp_var_309 = ((({ int _temp_var_310 = ((i % 4));
        (_temp_var_310 == 0 ? indices.field_0 : (_temp_var_310 == 1 ? indices.field_1 : (_temp_var_310 == 2 ? indices.field_2 : (_temp_var_310 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_309 == 0 ? indices.field_0 : (_temp_var_309 == 1 ? indices.field_1 : (_temp_var_309 == 2 ? indices.field_2 : (_temp_var_309 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_308 == 0 ? indices.field_0 : (_temp_var_308 == 1 ? indices.field_1 : (_temp_var_308 == 2 ? indices.field_2 : (_temp_var_308 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_209_ is already defined
#ifndef _block_k_209__func
#define _block_k_209__func
__device__ int _block_k_209_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_311 = ((({ int _temp_var_312 = ((({ int _temp_var_313 = ((i % 4));
        (_temp_var_313 == 0 ? indices.field_0 : (_temp_var_313 == 1 ? indices.field_1 : (_temp_var_313 == 2 ? indices.field_2 : (_temp_var_313 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_312 == 0 ? indices.field_0 : (_temp_var_312 == 1 ? indices.field_1 : (_temp_var_312 == 2 ? indices.field_2 : (_temp_var_312 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_311 == 0 ? indices.field_0 : (_temp_var_311 == 1 ? indices.field_1 : (_temp_var_311 == 2 ? indices.field_2 : (_temp_var_311 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_211_ is already defined
#ifndef _block_k_211__func
#define _block_k_211__func
__device__ int _block_k_211_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_314 = ((({ int _temp_var_315 = ((({ int _temp_var_316 = ((i % 4));
        (_temp_var_316 == 0 ? indices.field_0 : (_temp_var_316 == 1 ? indices.field_1 : (_temp_var_316 == 2 ? indices.field_2 : (_temp_var_316 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_315 == 0 ? indices.field_0 : (_temp_var_315 == 1 ? indices.field_1 : (_temp_var_315 == 2 ? indices.field_2 : (_temp_var_315 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_314 == 0 ? indices.field_0 : (_temp_var_314 == 1 ? indices.field_1 : (_temp_var_314 == 2 ? indices.field_2 : (_temp_var_314 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_213_ is already defined
#ifndef _block_k_213__func
#define _block_k_213__func
__device__ int _block_k_213_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_317 = ((({ int _temp_var_318 = ((({ int _temp_var_319 = ((i % 4));
        (_temp_var_319 == 0 ? indices.field_0 : (_temp_var_319 == 1 ? indices.field_1 : (_temp_var_319 == 2 ? indices.field_2 : (_temp_var_319 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_318 == 0 ? indices.field_0 : (_temp_var_318 == 1 ? indices.field_1 : (_temp_var_318 == 2 ? indices.field_2 : (_temp_var_318 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_317 == 0 ? indices.field_0 : (_temp_var_317 == 1 ? indices.field_1 : (_temp_var_317 == 2 ? indices.field_2 : (_temp_var_317 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_215_ is already defined
#ifndef _block_k_215__func
#define _block_k_215__func
__device__ int _block_k_215_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_320 = ((({ int _temp_var_321 = ((({ int _temp_var_322 = ((i % 4));
        (_temp_var_322 == 0 ? indices.field_0 : (_temp_var_322 == 1 ? indices.field_1 : (_temp_var_322 == 2 ? indices.field_2 : (_temp_var_322 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_321 == 0 ? indices.field_0 : (_temp_var_321 == 1 ? indices.field_1 : (_temp_var_321 == 2 ? indices.field_2 : (_temp_var_321 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_320 == 0 ? indices.field_0 : (_temp_var_320 == 1 ? indices.field_1 : (_temp_var_320 == 2 ? indices.field_2 : (_temp_var_320 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_217_ is already defined
#ifndef _block_k_217__func
#define _block_k_217__func
__device__ int _block_k_217_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_323 = ((({ int _temp_var_324 = ((({ int _temp_var_325 = ((i % 4));
        (_temp_var_325 == 0 ? indices.field_0 : (_temp_var_325 == 1 ? indices.field_1 : (_temp_var_325 == 2 ? indices.field_2 : (_temp_var_325 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_324 == 0 ? indices.field_0 : (_temp_var_324 == 1 ? indices.field_1 : (_temp_var_324 == 2 ? indices.field_2 : (_temp_var_324 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_323 == 0 ? indices.field_0 : (_temp_var_323 == 1 ? indices.field_1 : (_temp_var_323 == 2 ? indices.field_2 : (_temp_var_323 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_219_ is already defined
#ifndef _block_k_219__func
#define _block_k_219__func
__device__ int _block_k_219_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_326 = ((({ int _temp_var_327 = ((({ int _temp_var_328 = ((i % 4));
        (_temp_var_328 == 0 ? indices.field_0 : (_temp_var_328 == 1 ? indices.field_1 : (_temp_var_328 == 2 ? indices.field_2 : (_temp_var_328 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_327 == 0 ? indices.field_0 : (_temp_var_327 == 1 ? indices.field_1 : (_temp_var_327 == 2 ? indices.field_2 : (_temp_var_327 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_326 == 0 ? indices.field_0 : (_temp_var_326 == 1 ? indices.field_1 : (_temp_var_326 == 2 ? indices.field_2 : (_temp_var_326 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_221_ is already defined
#ifndef _block_k_221__func
#define _block_k_221__func
__device__ int _block_k_221_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_329 = ((({ int _temp_var_330 = ((({ int _temp_var_331 = ((i % 4));
        (_temp_var_331 == 0 ? indices.field_0 : (_temp_var_331 == 1 ? indices.field_1 : (_temp_var_331 == 2 ? indices.field_2 : (_temp_var_331 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_330 == 0 ? indices.field_0 : (_temp_var_330 == 1 ? indices.field_1 : (_temp_var_330 == 2 ? indices.field_2 : (_temp_var_330 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_329 == 0 ? indices.field_0 : (_temp_var_329 == 1 ? indices.field_1 : (_temp_var_329 == 2 ? indices.field_2 : (_temp_var_329 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_223_ is already defined
#ifndef _block_k_223__func
#define _block_k_223__func
__device__ int _block_k_223_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_332 = ((({ int _temp_var_333 = ((({ int _temp_var_334 = ((i % 4));
        (_temp_var_334 == 0 ? indices.field_0 : (_temp_var_334 == 1 ? indices.field_1 : (_temp_var_334 == 2 ? indices.field_2 : (_temp_var_334 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_333 == 0 ? indices.field_0 : (_temp_var_333 == 1 ? indices.field_1 : (_temp_var_333 == 2 ? indices.field_2 : (_temp_var_333 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_332 == 0 ? indices.field_0 : (_temp_var_332 == 1 ? indices.field_1 : (_temp_var_332 == 2 ? indices.field_2 : (_temp_var_332 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_225_ is already defined
#ifndef _block_k_225__func
#define _block_k_225__func
__device__ int _block_k_225_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_335 = ((({ int _temp_var_336 = ((({ int _temp_var_337 = ((i % 4));
        (_temp_var_337 == 0 ? indices.field_0 : (_temp_var_337 == 1 ? indices.field_1 : (_temp_var_337 == 2 ? indices.field_2 : (_temp_var_337 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_336 == 0 ? indices.field_0 : (_temp_var_336 == 1 ? indices.field_1 : (_temp_var_336 == 2 ? indices.field_2 : (_temp_var_336 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_335 == 0 ? indices.field_0 : (_temp_var_335 == 1 ? indices.field_1 : (_temp_var_335 == 2 ? indices.field_2 : (_temp_var_335 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_227_ is already defined
#ifndef _block_k_227__func
#define _block_k_227__func
__device__ int _block_k_227_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_338 = ((({ int _temp_var_339 = ((({ int _temp_var_340 = ((i % 4));
        (_temp_var_340 == 0 ? indices.field_0 : (_temp_var_340 == 1 ? indices.field_1 : (_temp_var_340 == 2 ? indices.field_2 : (_temp_var_340 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_339 == 0 ? indices.field_0 : (_temp_var_339 == 1 ? indices.field_1 : (_temp_var_339 == 2 ? indices.field_2 : (_temp_var_339 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_338 == 0 ? indices.field_0 : (_temp_var_338 == 1 ? indices.field_1 : (_temp_var_338 == 2 ? indices.field_2 : (_temp_var_338 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_229_ is already defined
#ifndef _block_k_229__func
#define _block_k_229__func
__device__ int _block_k_229_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_341 = ((({ int _temp_var_342 = ((({ int _temp_var_343 = ((i % 4));
        (_temp_var_343 == 0 ? indices.field_0 : (_temp_var_343 == 1 ? indices.field_1 : (_temp_var_343 == 2 ? indices.field_2 : (_temp_var_343 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_342 == 0 ? indices.field_0 : (_temp_var_342 == 1 ? indices.field_1 : (_temp_var_342 == 2 ? indices.field_2 : (_temp_var_342 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_341 == 0 ? indices.field_0 : (_temp_var_341 == 1 ? indices.field_1 : (_temp_var_341 == 2 ? indices.field_2 : (_temp_var_341 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_231_ is already defined
#ifndef _block_k_231__func
#define _block_k_231__func
__device__ int _block_k_231_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_344 = ((({ int _temp_var_345 = ((({ int _temp_var_346 = ((i % 4));
        (_temp_var_346 == 0 ? indices.field_0 : (_temp_var_346 == 1 ? indices.field_1 : (_temp_var_346 == 2 ? indices.field_2 : (_temp_var_346 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_345 == 0 ? indices.field_0 : (_temp_var_345 == 1 ? indices.field_1 : (_temp_var_345 == 2 ? indices.field_2 : (_temp_var_345 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_344 == 0 ? indices.field_0 : (_temp_var_344 == 1 ? indices.field_1 : (_temp_var_344 == 2 ? indices.field_2 : (_temp_var_344 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_233_ is already defined
#ifndef _block_k_233__func
#define _block_k_233__func
__device__ int _block_k_233_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_347 = ((({ int _temp_var_348 = ((({ int _temp_var_349 = ((i % 4));
        (_temp_var_349 == 0 ? indices.field_0 : (_temp_var_349 == 1 ? indices.field_1 : (_temp_var_349 == 2 ? indices.field_2 : (_temp_var_349 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_348 == 0 ? indices.field_0 : (_temp_var_348 == 1 ? indices.field_1 : (_temp_var_348 == 2 ? indices.field_2 : (_temp_var_348 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_347 == 0 ? indices.field_0 : (_temp_var_347 == 1 ? indices.field_1 : (_temp_var_347 == 2 ? indices.field_2 : (_temp_var_347 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_235_ is already defined
#ifndef _block_k_235__func
#define _block_k_235__func
__device__ int _block_k_235_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_350 = ((({ int _temp_var_351 = ((({ int _temp_var_352 = ((i % 4));
        (_temp_var_352 == 0 ? indices.field_0 : (_temp_var_352 == 1 ? indices.field_1 : (_temp_var_352 == 2 ? indices.field_2 : (_temp_var_352 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_351 == 0 ? indices.field_0 : (_temp_var_351 == 1 ? indices.field_1 : (_temp_var_351 == 2 ? indices.field_2 : (_temp_var_351 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_350 == 0 ? indices.field_0 : (_temp_var_350 == 1 ? indices.field_1 : (_temp_var_350 == 2 ? indices.field_2 : (_temp_var_350 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_237_ is already defined
#ifndef _block_k_237__func
#define _block_k_237__func
__device__ int _block_k_237_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_353 = ((({ int _temp_var_354 = ((({ int _temp_var_355 = ((i % 4));
        (_temp_var_355 == 0 ? indices.field_0 : (_temp_var_355 == 1 ? indices.field_1 : (_temp_var_355 == 2 ? indices.field_2 : (_temp_var_355 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_354 == 0 ? indices.field_0 : (_temp_var_354 == 1 ? indices.field_1 : (_temp_var_354 == 2 ? indices.field_2 : (_temp_var_354 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_353 == 0 ? indices.field_0 : (_temp_var_353 == 1 ? indices.field_1 : (_temp_var_353 == 2 ? indices.field_2 : (_temp_var_353 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_239_ is already defined
#ifndef _block_k_239__func
#define _block_k_239__func
__device__ int _block_k_239_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_356 = ((({ int _temp_var_357 = ((({ int _temp_var_358 = ((i % 4));
        (_temp_var_358 == 0 ? indices.field_0 : (_temp_var_358 == 1 ? indices.field_1 : (_temp_var_358 == 2 ? indices.field_2 : (_temp_var_358 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_357 == 0 ? indices.field_0 : (_temp_var_357 == 1 ? indices.field_1 : (_temp_var_357 == 2 ? indices.field_2 : (_temp_var_357 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_356 == 0 ? indices.field_0 : (_temp_var_356 == 1 ? indices.field_1 : (_temp_var_356 == 2 ? indices.field_2 : (_temp_var_356 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_241_ is already defined
#ifndef _block_k_241__func
#define _block_k_241__func
__device__ int _block_k_241_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_359 = ((({ int _temp_var_360 = ((({ int _temp_var_361 = ((i % 4));
        (_temp_var_361 == 0 ? indices.field_0 : (_temp_var_361 == 1 ? indices.field_1 : (_temp_var_361 == 2 ? indices.field_2 : (_temp_var_361 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_360 == 0 ? indices.field_0 : (_temp_var_360 == 1 ? indices.field_1 : (_temp_var_360 == 2 ? indices.field_2 : (_temp_var_360 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_359 == 0 ? indices.field_0 : (_temp_var_359 == 1 ? indices.field_1 : (_temp_var_359 == 2 ? indices.field_2 : (_temp_var_359 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_243_ is already defined
#ifndef _block_k_243__func
#define _block_k_243__func
__device__ int _block_k_243_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_362 = ((({ int _temp_var_363 = ((({ int _temp_var_364 = ((i % 4));
        (_temp_var_364 == 0 ? indices.field_0 : (_temp_var_364 == 1 ? indices.field_1 : (_temp_var_364 == 2 ? indices.field_2 : (_temp_var_364 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_363 == 0 ? indices.field_0 : (_temp_var_363 == 1 ? indices.field_1 : (_temp_var_363 == 2 ? indices.field_2 : (_temp_var_363 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_362 == 0 ? indices.field_0 : (_temp_var_362 == 1 ? indices.field_1 : (_temp_var_362 == 2 ? indices.field_2 : (_temp_var_362 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_245_ is already defined
#ifndef _block_k_245__func
#define _block_k_245__func
__device__ int _block_k_245_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_365 = ((({ int _temp_var_366 = ((({ int _temp_var_367 = ((i % 4));
        (_temp_var_367 == 0 ? indices.field_0 : (_temp_var_367 == 1 ? indices.field_1 : (_temp_var_367 == 2 ? indices.field_2 : (_temp_var_367 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_366 == 0 ? indices.field_0 : (_temp_var_366 == 1 ? indices.field_1 : (_temp_var_366 == 2 ? indices.field_2 : (_temp_var_366 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_365 == 0 ? indices.field_0 : (_temp_var_365 == 1 ? indices.field_1 : (_temp_var_365 == 2 ? indices.field_2 : (_temp_var_365 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_247_ is already defined
#ifndef _block_k_247__func
#define _block_k_247__func
__device__ int _block_k_247_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_368 = ((({ int _temp_var_369 = ((({ int _temp_var_370 = ((i % 4));
        (_temp_var_370 == 0 ? indices.field_0 : (_temp_var_370 == 1 ? indices.field_1 : (_temp_var_370 == 2 ? indices.field_2 : (_temp_var_370 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_369 == 0 ? indices.field_0 : (_temp_var_369 == 1 ? indices.field_1 : (_temp_var_369 == 2 ? indices.field_2 : (_temp_var_369 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_368 == 0 ? indices.field_0 : (_temp_var_368 == 1 ? indices.field_1 : (_temp_var_368 == 2 ? indices.field_2 : (_temp_var_368 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_249_ is already defined
#ifndef _block_k_249__func
#define _block_k_249__func
__device__ int _block_k_249_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_371 = ((({ int _temp_var_372 = ((({ int _temp_var_373 = ((i % 4));
        (_temp_var_373 == 0 ? indices.field_0 : (_temp_var_373 == 1 ? indices.field_1 : (_temp_var_373 == 2 ? indices.field_2 : (_temp_var_373 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_372 == 0 ? indices.field_0 : (_temp_var_372 == 1 ? indices.field_1 : (_temp_var_372 == 2 ? indices.field_2 : (_temp_var_372 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_371 == 0 ? indices.field_0 : (_temp_var_371 == 1 ? indices.field_1 : (_temp_var_371 == 2 ? indices.field_2 : (_temp_var_371 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_251_ is already defined
#ifndef _block_k_251__func
#define _block_k_251__func
__device__ int _block_k_251_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_374 = ((({ int _temp_var_375 = ((({ int _temp_var_376 = ((i % 4));
        (_temp_var_376 == 0 ? indices.field_0 : (_temp_var_376 == 1 ? indices.field_1 : (_temp_var_376 == 2 ? indices.field_2 : (_temp_var_376 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_375 == 0 ? indices.field_0 : (_temp_var_375 == 1 ? indices.field_1 : (_temp_var_375 == 2 ? indices.field_2 : (_temp_var_375 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_374 == 0 ? indices.field_0 : (_temp_var_374 == 1 ? indices.field_1 : (_temp_var_374 == 2 ? indices.field_2 : (_temp_var_374 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_253_ is already defined
#ifndef _block_k_253__func
#define _block_k_253__func
__device__ int _block_k_253_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_377 = ((({ int _temp_var_378 = ((({ int _temp_var_379 = ((i % 4));
        (_temp_var_379 == 0 ? indices.field_0 : (_temp_var_379 == 1 ? indices.field_1 : (_temp_var_379 == 2 ? indices.field_2 : (_temp_var_379 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_378 == 0 ? indices.field_0 : (_temp_var_378 == 1 ? indices.field_1 : (_temp_var_378 == 2 ? indices.field_2 : (_temp_var_378 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_377 == 0 ? indices.field_0 : (_temp_var_377 == 1 ? indices.field_1 : (_temp_var_377 == 2 ? indices.field_2 : (_temp_var_377 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_255_ is already defined
#ifndef _block_k_255__func
#define _block_k_255__func
__device__ int _block_k_255_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_380 = ((({ int _temp_var_381 = ((({ int _temp_var_382 = ((i % 4));
        (_temp_var_382 == 0 ? indices.field_0 : (_temp_var_382 == 1 ? indices.field_1 : (_temp_var_382 == 2 ? indices.field_2 : (_temp_var_382 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_381 == 0 ? indices.field_0 : (_temp_var_381 == 1 ? indices.field_1 : (_temp_var_381 == 2 ? indices.field_2 : (_temp_var_381 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_380 == 0 ? indices.field_0 : (_temp_var_380 == 1 ? indices.field_1 : (_temp_var_380 == 2 ? indices.field_2 : (_temp_var_380 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_257_ is already defined
#ifndef _block_k_257__func
#define _block_k_257__func
__device__ int _block_k_257_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_383 = ((({ int _temp_var_384 = ((({ int _temp_var_385 = ((i % 4));
        (_temp_var_385 == 0 ? indices.field_0 : (_temp_var_385 == 1 ? indices.field_1 : (_temp_var_385 == 2 ? indices.field_2 : (_temp_var_385 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_384 == 0 ? indices.field_0 : (_temp_var_384 == 1 ? indices.field_1 : (_temp_var_384 == 2 ? indices.field_2 : (_temp_var_384 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_383 == 0 ? indices.field_0 : (_temp_var_383 == 1 ? indices.field_1 : (_temp_var_383 == 2 ? indices.field_2 : (_temp_var_383 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_259_ is already defined
#ifndef _block_k_259__func
#define _block_k_259__func
__device__ int _block_k_259_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_386 = ((({ int _temp_var_387 = ((({ int _temp_var_388 = ((i % 4));
        (_temp_var_388 == 0 ? indices.field_0 : (_temp_var_388 == 1 ? indices.field_1 : (_temp_var_388 == 2 ? indices.field_2 : (_temp_var_388 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_387 == 0 ? indices.field_0 : (_temp_var_387 == 1 ? indices.field_1 : (_temp_var_387 == 2 ? indices.field_2 : (_temp_var_387 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_386 == 0 ? indices.field_0 : (_temp_var_386 == 1 ? indices.field_1 : (_temp_var_386 == 2 ? indices.field_2 : (_temp_var_386 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_261_ is already defined
#ifndef _block_k_261__func
#define _block_k_261__func
__device__ int _block_k_261_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_389 = ((({ int _temp_var_390 = ((({ int _temp_var_391 = ((i % 4));
        (_temp_var_391 == 0 ? indices.field_0 : (_temp_var_391 == 1 ? indices.field_1 : (_temp_var_391 == 2 ? indices.field_2 : (_temp_var_391 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_390 == 0 ? indices.field_0 : (_temp_var_390 == 1 ? indices.field_1 : (_temp_var_390 == 2 ? indices.field_2 : (_temp_var_390 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_389 == 0 ? indices.field_0 : (_temp_var_389 == 1 ? indices.field_1 : (_temp_var_389 == 2 ? indices.field_2 : (_temp_var_389 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_263_ is already defined
#ifndef _block_k_263__func
#define _block_k_263__func
__device__ int _block_k_263_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_392 = ((({ int _temp_var_393 = ((({ int _temp_var_394 = ((i % 4));
        (_temp_var_394 == 0 ? indices.field_0 : (_temp_var_394 == 1 ? indices.field_1 : (_temp_var_394 == 2 ? indices.field_2 : (_temp_var_394 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_393 == 0 ? indices.field_0 : (_temp_var_393 == 1 ? indices.field_1 : (_temp_var_393 == 2 ? indices.field_2 : (_temp_var_393 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_392 == 0 ? indices.field_0 : (_temp_var_392 == 1 ? indices.field_1 : (_temp_var_392 == 2 ? indices.field_2 : (_temp_var_392 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_265_ is already defined
#ifndef _block_k_265__func
#define _block_k_265__func
__device__ int _block_k_265_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_395 = ((({ int _temp_var_396 = ((({ int _temp_var_397 = ((i % 4));
        (_temp_var_397 == 0 ? indices.field_0 : (_temp_var_397 == 1 ? indices.field_1 : (_temp_var_397 == 2 ? indices.field_2 : (_temp_var_397 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_396 == 0 ? indices.field_0 : (_temp_var_396 == 1 ? indices.field_1 : (_temp_var_396 == 2 ? indices.field_2 : (_temp_var_396 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_395 == 0 ? indices.field_0 : (_temp_var_395 == 1 ? indices.field_1 : (_temp_var_395 == 2 ? indices.field_2 : (_temp_var_395 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_267_ is already defined
#ifndef _block_k_267__func
#define _block_k_267__func
__device__ int _block_k_267_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_398 = ((({ int _temp_var_399 = ((({ int _temp_var_400 = ((i % 4));
        (_temp_var_400 == 0 ? indices.field_0 : (_temp_var_400 == 1 ? indices.field_1 : (_temp_var_400 == 2 ? indices.field_2 : (_temp_var_400 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_399 == 0 ? indices.field_0 : (_temp_var_399 == 1 ? indices.field_1 : (_temp_var_399 == 2 ? indices.field_2 : (_temp_var_399 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_398 == 0 ? indices.field_0 : (_temp_var_398 == 1 ? indices.field_1 : (_temp_var_398 == 2 ? indices.field_2 : (_temp_var_398 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_269_ is already defined
#ifndef _block_k_269__func
#define _block_k_269__func
__device__ int _block_k_269_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_401 = ((({ int _temp_var_402 = ((({ int _temp_var_403 = ((i % 4));
        (_temp_var_403 == 0 ? indices.field_0 : (_temp_var_403 == 1 ? indices.field_1 : (_temp_var_403 == 2 ? indices.field_2 : (_temp_var_403 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_402 == 0 ? indices.field_0 : (_temp_var_402 == 1 ? indices.field_1 : (_temp_var_402 == 2 ? indices.field_2 : (_temp_var_402 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_401 == 0 ? indices.field_0 : (_temp_var_401 == 1 ? indices.field_1 : (_temp_var_401 == 2 ? indices.field_2 : (_temp_var_401 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_271_ is already defined
#ifndef _block_k_271__func
#define _block_k_271__func
__device__ int _block_k_271_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_404 = ((({ int _temp_var_405 = ((({ int _temp_var_406 = ((i % 4));
        (_temp_var_406 == 0 ? indices.field_0 : (_temp_var_406 == 1 ? indices.field_1 : (_temp_var_406 == 2 ? indices.field_2 : (_temp_var_406 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_405 == 0 ? indices.field_0 : (_temp_var_405 == 1 ? indices.field_1 : (_temp_var_405 == 2 ? indices.field_2 : (_temp_var_405 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_404 == 0 ? indices.field_0 : (_temp_var_404 == 1 ? indices.field_1 : (_temp_var_404 == 2 ? indices.field_2 : (_temp_var_404 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_273_ is already defined
#ifndef _block_k_273__func
#define _block_k_273__func
__device__ int _block_k_273_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_407 = ((({ int _temp_var_408 = ((({ int _temp_var_409 = ((i % 4));
        (_temp_var_409 == 0 ? indices.field_0 : (_temp_var_409 == 1 ? indices.field_1 : (_temp_var_409 == 2 ? indices.field_2 : (_temp_var_409 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_408 == 0 ? indices.field_0 : (_temp_var_408 == 1 ? indices.field_1 : (_temp_var_408 == 2 ? indices.field_2 : (_temp_var_408 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_407 == 0 ? indices.field_0 : (_temp_var_407 == 1 ? indices.field_1 : (_temp_var_407 == 2 ? indices.field_2 : (_temp_var_407 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_275_ is already defined
#ifndef _block_k_275__func
#define _block_k_275__func
__device__ int _block_k_275_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_410 = ((({ int _temp_var_411 = ((({ int _temp_var_412 = ((i % 4));
        (_temp_var_412 == 0 ? indices.field_0 : (_temp_var_412 == 1 ? indices.field_1 : (_temp_var_412 == 2 ? indices.field_2 : (_temp_var_412 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_411 == 0 ? indices.field_0 : (_temp_var_411 == 1 ? indices.field_1 : (_temp_var_411 == 2 ? indices.field_2 : (_temp_var_411 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_410 == 0 ? indices.field_0 : (_temp_var_410 == 1 ? indices.field_1 : (_temp_var_410 == 2 ? indices.field_2 : (_temp_var_410 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_277_ is already defined
#ifndef _block_k_277__func
#define _block_k_277__func
__device__ int _block_k_277_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_413 = ((({ int _temp_var_414 = ((({ int _temp_var_415 = ((i % 4));
        (_temp_var_415 == 0 ? indices.field_0 : (_temp_var_415 == 1 ? indices.field_1 : (_temp_var_415 == 2 ? indices.field_2 : (_temp_var_415 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_414 == 0 ? indices.field_0 : (_temp_var_414 == 1 ? indices.field_1 : (_temp_var_414 == 2 ? indices.field_2 : (_temp_var_414 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_413 == 0 ? indices.field_0 : (_temp_var_413 == 1 ? indices.field_1 : (_temp_var_413 == 2 ? indices.field_2 : (_temp_var_413 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_279_ is already defined
#ifndef _block_k_279__func
#define _block_k_279__func
__device__ int _block_k_279_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_416 = ((({ int _temp_var_417 = ((({ int _temp_var_418 = ((i % 4));
        (_temp_var_418 == 0 ? indices.field_0 : (_temp_var_418 == 1 ? indices.field_1 : (_temp_var_418 == 2 ? indices.field_2 : (_temp_var_418 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_417 == 0 ? indices.field_0 : (_temp_var_417 == 1 ? indices.field_1 : (_temp_var_417 == 2 ? indices.field_2 : (_temp_var_417 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_416 == 0 ? indices.field_0 : (_temp_var_416 == 1 ? indices.field_1 : (_temp_var_416 == 2 ? indices.field_2 : (_temp_var_416 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_281_ is already defined
#ifndef _block_k_281__func
#define _block_k_281__func
__device__ int _block_k_281_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_419 = ((({ int _temp_var_420 = ((({ int _temp_var_421 = ((i % 4));
        (_temp_var_421 == 0 ? indices.field_0 : (_temp_var_421 == 1 ? indices.field_1 : (_temp_var_421 == 2 ? indices.field_2 : (_temp_var_421 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_420 == 0 ? indices.field_0 : (_temp_var_420 == 1 ? indices.field_1 : (_temp_var_420 == 2 ? indices.field_2 : (_temp_var_420 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_419 == 0 ? indices.field_0 : (_temp_var_419 == 1 ? indices.field_1 : (_temp_var_419 == 2 ? indices.field_2 : (_temp_var_419 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_283_ is already defined
#ifndef _block_k_283__func
#define _block_k_283__func
__device__ int _block_k_283_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_422 = ((({ int _temp_var_423 = ((({ int _temp_var_424 = ((i % 4));
        (_temp_var_424 == 0 ? indices.field_0 : (_temp_var_424 == 1 ? indices.field_1 : (_temp_var_424 == 2 ? indices.field_2 : (_temp_var_424 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_423 == 0 ? indices.field_0 : (_temp_var_423 == 1 ? indices.field_1 : (_temp_var_423 == 2 ? indices.field_2 : (_temp_var_423 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_422 == 0 ? indices.field_0 : (_temp_var_422 == 1 ? indices.field_1 : (_temp_var_422 == 2 ? indices.field_2 : (_temp_var_422 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_285_ is already defined
#ifndef _block_k_285__func
#define _block_k_285__func
__device__ int _block_k_285_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_425 = ((({ int _temp_var_426 = ((({ int _temp_var_427 = ((i % 4));
        (_temp_var_427 == 0 ? indices.field_0 : (_temp_var_427 == 1 ? indices.field_1 : (_temp_var_427 == 2 ? indices.field_2 : (_temp_var_427 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_426 == 0 ? indices.field_0 : (_temp_var_426 == 1 ? indices.field_1 : (_temp_var_426 == 2 ? indices.field_2 : (_temp_var_426 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_425 == 0 ? indices.field_0 : (_temp_var_425 == 1 ? indices.field_1 : (_temp_var_425 == 2 ? indices.field_2 : (_temp_var_425 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_287_ is already defined
#ifndef _block_k_287__func
#define _block_k_287__func
__device__ int _block_k_287_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_428 = ((({ int _temp_var_429 = ((({ int _temp_var_430 = ((i % 4));
        (_temp_var_430 == 0 ? indices.field_0 : (_temp_var_430 == 1 ? indices.field_1 : (_temp_var_430 == 2 ? indices.field_2 : (_temp_var_430 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_429 == 0 ? indices.field_0 : (_temp_var_429 == 1 ? indices.field_1 : (_temp_var_429 == 2 ? indices.field_2 : (_temp_var_429 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_428 == 0 ? indices.field_0 : (_temp_var_428 == 1 ? indices.field_1 : (_temp_var_428 == 2 ? indices.field_2 : (_temp_var_428 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_289_ is already defined
#ifndef _block_k_289__func
#define _block_k_289__func
__device__ int _block_k_289_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_431 = ((({ int _temp_var_432 = ((({ int _temp_var_433 = ((i % 4));
        (_temp_var_433 == 0 ? indices.field_0 : (_temp_var_433 == 1 ? indices.field_1 : (_temp_var_433 == 2 ? indices.field_2 : (_temp_var_433 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_432 == 0 ? indices.field_0 : (_temp_var_432 == 1 ? indices.field_1 : (_temp_var_432 == 2 ? indices.field_2 : (_temp_var_432 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_431 == 0 ? indices.field_0 : (_temp_var_431 == 1 ? indices.field_1 : (_temp_var_431 == 2 ? indices.field_2 : (_temp_var_431 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_291_ is already defined
#ifndef _block_k_291__func
#define _block_k_291__func
__device__ int _block_k_291_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_434 = ((({ int _temp_var_435 = ((({ int _temp_var_436 = ((i % 4));
        (_temp_var_436 == 0 ? indices.field_0 : (_temp_var_436 == 1 ? indices.field_1 : (_temp_var_436 == 2 ? indices.field_2 : (_temp_var_436 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_435 == 0 ? indices.field_0 : (_temp_var_435 == 1 ? indices.field_1 : (_temp_var_435 == 2 ? indices.field_2 : (_temp_var_435 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_434 == 0 ? indices.field_0 : (_temp_var_434 == 1 ? indices.field_1 : (_temp_var_434 == 2 ? indices.field_2 : (_temp_var_434 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_293_ is already defined
#ifndef _block_k_293__func
#define _block_k_293__func
__device__ int _block_k_293_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_437 = ((({ int _temp_var_438 = ((({ int _temp_var_439 = ((i % 4));
        (_temp_var_439 == 0 ? indices.field_0 : (_temp_var_439 == 1 ? indices.field_1 : (_temp_var_439 == 2 ? indices.field_2 : (_temp_var_439 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_438 == 0 ? indices.field_0 : (_temp_var_438 == 1 ? indices.field_1 : (_temp_var_438 == 2 ? indices.field_2 : (_temp_var_438 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_437 == 0 ? indices.field_0 : (_temp_var_437 == 1 ? indices.field_1 : (_temp_var_437 == 2 ? indices.field_2 : (_temp_var_437 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_295_ is already defined
#ifndef _block_k_295__func
#define _block_k_295__func
__device__ int _block_k_295_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_440 = ((({ int _temp_var_441 = ((({ int _temp_var_442 = ((i % 4));
        (_temp_var_442 == 0 ? indices.field_0 : (_temp_var_442 == 1 ? indices.field_1 : (_temp_var_442 == 2 ? indices.field_2 : (_temp_var_442 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_441 == 0 ? indices.field_0 : (_temp_var_441 == 1 ? indices.field_1 : (_temp_var_441 == 2 ? indices.field_2 : (_temp_var_441 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_440 == 0 ? indices.field_0 : (_temp_var_440 == 1 ? indices.field_1 : (_temp_var_440 == 2 ? indices.field_2 : (_temp_var_440 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_297_ is already defined
#ifndef _block_k_297__func
#define _block_k_297__func
__device__ int _block_k_297_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_443 = ((({ int _temp_var_444 = ((({ int _temp_var_445 = ((i % 4));
        (_temp_var_445 == 0 ? indices.field_0 : (_temp_var_445 == 1 ? indices.field_1 : (_temp_var_445 == 2 ? indices.field_2 : (_temp_var_445 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_444 == 0 ? indices.field_0 : (_temp_var_444 == 1 ? indices.field_1 : (_temp_var_444 == 2 ? indices.field_2 : (_temp_var_444 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_443 == 0 ? indices.field_0 : (_temp_var_443 == 1 ? indices.field_1 : (_temp_var_443 == 2 ? indices.field_2 : (_temp_var_443 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_299_ is already defined
#ifndef _block_k_299__func
#define _block_k_299__func
__device__ int _block_k_299_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_446 = ((({ int _temp_var_447 = ((({ int _temp_var_448 = ((i % 4));
        (_temp_var_448 == 0 ? indices.field_0 : (_temp_var_448 == 1 ? indices.field_1 : (_temp_var_448 == 2 ? indices.field_2 : (_temp_var_448 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_447 == 0 ? indices.field_0 : (_temp_var_447 == 1 ? indices.field_1 : (_temp_var_447 == 2 ? indices.field_2 : (_temp_var_447 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_446 == 0 ? indices.field_0 : (_temp_var_446 == 1 ? indices.field_1 : (_temp_var_446 == 2 ? indices.field_2 : (_temp_var_446 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_301_ is already defined
#ifndef _block_k_301__func
#define _block_k_301__func
__device__ int _block_k_301_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_449 = ((({ int _temp_var_450 = ((({ int _temp_var_451 = ((i % 4));
        (_temp_var_451 == 0 ? indices.field_0 : (_temp_var_451 == 1 ? indices.field_1 : (_temp_var_451 == 2 ? indices.field_2 : (_temp_var_451 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_450 == 0 ? indices.field_0 : (_temp_var_450 == 1 ? indices.field_1 : (_temp_var_450 == 2 ? indices.field_2 : (_temp_var_450 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_449 == 0 ? indices.field_0 : (_temp_var_449 == 1 ? indices.field_1 : (_temp_var_449 == 2 ? indices.field_2 : (_temp_var_449 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_303_ is already defined
#ifndef _block_k_303__func
#define _block_k_303__func
__device__ int _block_k_303_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_452 = ((({ int _temp_var_453 = ((({ int _temp_var_454 = ((i % 4));
        (_temp_var_454 == 0 ? indices.field_0 : (_temp_var_454 == 1 ? indices.field_1 : (_temp_var_454 == 2 ? indices.field_2 : (_temp_var_454 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_453 == 0 ? indices.field_0 : (_temp_var_453 == 1 ? indices.field_1 : (_temp_var_453 == 2 ? indices.field_2 : (_temp_var_453 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_452 == 0 ? indices.field_0 : (_temp_var_452 == 1 ? indices.field_1 : (_temp_var_452 == 2 ? indices.field_2 : (_temp_var_452 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_305_ is already defined
#ifndef _block_k_305__func
#define _block_k_305__func
__device__ int _block_k_305_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_455 = ((({ int _temp_var_456 = ((({ int _temp_var_457 = ((i % 4));
        (_temp_var_457 == 0 ? indices.field_0 : (_temp_var_457 == 1 ? indices.field_1 : (_temp_var_457 == 2 ? indices.field_2 : (_temp_var_457 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_456 == 0 ? indices.field_0 : (_temp_var_456 == 1 ? indices.field_1 : (_temp_var_456 == 2 ? indices.field_2 : (_temp_var_456 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_455 == 0 ? indices.field_0 : (_temp_var_455 == 1 ? indices.field_1 : (_temp_var_455 == 2 ? indices.field_2 : (_temp_var_455 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_307_ is already defined
#ifndef _block_k_307__func
#define _block_k_307__func
__device__ int _block_k_307_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_458 = ((({ int _temp_var_459 = ((({ int _temp_var_460 = ((i % 4));
        (_temp_var_460 == 0 ? indices.field_0 : (_temp_var_460 == 1 ? indices.field_1 : (_temp_var_460 == 2 ? indices.field_2 : (_temp_var_460 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_459 == 0 ? indices.field_0 : (_temp_var_459 == 1 ? indices.field_1 : (_temp_var_459 == 2 ? indices.field_2 : (_temp_var_459 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_458 == 0 ? indices.field_0 : (_temp_var_458 == 1 ? indices.field_1 : (_temp_var_458 == 2 ? indices.field_2 : (_temp_var_458 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_309_ is already defined
#ifndef _block_k_309__func
#define _block_k_309__func
__device__ int _block_k_309_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_461 = ((({ int _temp_var_462 = ((({ int _temp_var_463 = ((i % 4));
        (_temp_var_463 == 0 ? indices.field_0 : (_temp_var_463 == 1 ? indices.field_1 : (_temp_var_463 == 2 ? indices.field_2 : (_temp_var_463 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_462 == 0 ? indices.field_0 : (_temp_var_462 == 1 ? indices.field_1 : (_temp_var_462 == 2 ? indices.field_2 : (_temp_var_462 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_461 == 0 ? indices.field_0 : (_temp_var_461 == 1 ? indices.field_1 : (_temp_var_461 == 2 ? indices.field_2 : (_temp_var_461 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_311_ is already defined
#ifndef _block_k_311__func
#define _block_k_311__func
__device__ int _block_k_311_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_464 = ((({ int _temp_var_465 = ((({ int _temp_var_466 = ((i % 4));
        (_temp_var_466 == 0 ? indices.field_0 : (_temp_var_466 == 1 ? indices.field_1 : (_temp_var_466 == 2 ? indices.field_2 : (_temp_var_466 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_465 == 0 ? indices.field_0 : (_temp_var_465 == 1 ? indices.field_1 : (_temp_var_465 == 2 ? indices.field_2 : (_temp_var_465 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_464 == 0 ? indices.field_0 : (_temp_var_464 == 1 ? indices.field_1 : (_temp_var_464 == 2 ? indices.field_2 : (_temp_var_464 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_313_ is already defined
#ifndef _block_k_313__func
#define _block_k_313__func
__device__ int _block_k_313_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_467 = ((({ int _temp_var_468 = ((({ int _temp_var_469 = ((i % 4));
        (_temp_var_469 == 0 ? indices.field_0 : (_temp_var_469 == 1 ? indices.field_1 : (_temp_var_469 == 2 ? indices.field_2 : (_temp_var_469 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_468 == 0 ? indices.field_0 : (_temp_var_468 == 1 ? indices.field_1 : (_temp_var_468 == 2 ? indices.field_2 : (_temp_var_468 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_467 == 0 ? indices.field_0 : (_temp_var_467 == 1 ? indices.field_1 : (_temp_var_467 == 2 ? indices.field_2 : (_temp_var_467 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_315_ is already defined
#ifndef _block_k_315__func
#define _block_k_315__func
__device__ int _block_k_315_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_470 = ((({ int _temp_var_471 = ((({ int _temp_var_472 = ((i % 4));
        (_temp_var_472 == 0 ? indices.field_0 : (_temp_var_472 == 1 ? indices.field_1 : (_temp_var_472 == 2 ? indices.field_2 : (_temp_var_472 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_471 == 0 ? indices.field_0 : (_temp_var_471 == 1 ? indices.field_1 : (_temp_var_471 == 2 ? indices.field_2 : (_temp_var_471 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_470 == 0 ? indices.field_0 : (_temp_var_470 == 1 ? indices.field_1 : (_temp_var_470 == 2 ? indices.field_2 : (_temp_var_470 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_317_ is already defined
#ifndef _block_k_317__func
#define _block_k_317__func
__device__ int _block_k_317_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_473 = ((({ int _temp_var_474 = ((({ int _temp_var_475 = ((i % 4));
        (_temp_var_475 == 0 ? indices.field_0 : (_temp_var_475 == 1 ? indices.field_1 : (_temp_var_475 == 2 ? indices.field_2 : (_temp_var_475 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_474 == 0 ? indices.field_0 : (_temp_var_474 == 1 ? indices.field_1 : (_temp_var_474 == 2 ? indices.field_2 : (_temp_var_474 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_473 == 0 ? indices.field_0 : (_temp_var_473 == 1 ? indices.field_1 : (_temp_var_473 == 2 ? indices.field_2 : (_temp_var_473 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_319_ is already defined
#ifndef _block_k_319__func
#define _block_k_319__func
__device__ int _block_k_319_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_476 = ((({ int _temp_var_477 = ((({ int _temp_var_478 = ((i % 4));
        (_temp_var_478 == 0 ? indices.field_0 : (_temp_var_478 == 1 ? indices.field_1 : (_temp_var_478 == 2 ? indices.field_2 : (_temp_var_478 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_477 == 0 ? indices.field_0 : (_temp_var_477 == 1 ? indices.field_1 : (_temp_var_477 == 2 ? indices.field_2 : (_temp_var_477 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_476 == 0 ? indices.field_0 : (_temp_var_476 == 1 ? indices.field_1 : (_temp_var_476 == 2 ? indices.field_2 : (_temp_var_476 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_321_ is already defined
#ifndef _block_k_321__func
#define _block_k_321__func
__device__ int _block_k_321_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_479 = ((({ int _temp_var_480 = ((({ int _temp_var_481 = ((i % 4));
        (_temp_var_481 == 0 ? indices.field_0 : (_temp_var_481 == 1 ? indices.field_1 : (_temp_var_481 == 2 ? indices.field_2 : (_temp_var_481 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_480 == 0 ? indices.field_0 : (_temp_var_480 == 1 ? indices.field_1 : (_temp_var_480 == 2 ? indices.field_2 : (_temp_var_480 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_479 == 0 ? indices.field_0 : (_temp_var_479 == 1 ? indices.field_1 : (_temp_var_479 == 2 ? indices.field_2 : (_temp_var_479 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_323_ is already defined
#ifndef _block_k_323__func
#define _block_k_323__func
__device__ int _block_k_323_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_482 = ((({ int _temp_var_483 = ((({ int _temp_var_484 = ((i % 4));
        (_temp_var_484 == 0 ? indices.field_0 : (_temp_var_484 == 1 ? indices.field_1 : (_temp_var_484 == 2 ? indices.field_2 : (_temp_var_484 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_483 == 0 ? indices.field_0 : (_temp_var_483 == 1 ? indices.field_1 : (_temp_var_483 == 2 ? indices.field_2 : (_temp_var_483 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_482 == 0 ? indices.field_0 : (_temp_var_482 == 1 ? indices.field_1 : (_temp_var_482 == 2 ? indices.field_2 : (_temp_var_482 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_325_ is already defined
#ifndef _block_k_325__func
#define _block_k_325__func
__device__ int _block_k_325_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_485 = ((({ int _temp_var_486 = ((({ int _temp_var_487 = ((i % 4));
        (_temp_var_487 == 0 ? indices.field_0 : (_temp_var_487 == 1 ? indices.field_1 : (_temp_var_487 == 2 ? indices.field_2 : (_temp_var_487 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_486 == 0 ? indices.field_0 : (_temp_var_486 == 1 ? indices.field_1 : (_temp_var_486 == 2 ? indices.field_2 : (_temp_var_486 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_485 == 0 ? indices.field_0 : (_temp_var_485 == 1 ? indices.field_1 : (_temp_var_485 == 2 ? indices.field_2 : (_temp_var_485 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_327_ is already defined
#ifndef _block_k_327__func
#define _block_k_327__func
__device__ int _block_k_327_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_488 = ((({ int _temp_var_489 = ((({ int _temp_var_490 = ((i % 4));
        (_temp_var_490 == 0 ? indices.field_0 : (_temp_var_490 == 1 ? indices.field_1 : (_temp_var_490 == 2 ? indices.field_2 : (_temp_var_490 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_489 == 0 ? indices.field_0 : (_temp_var_489 == 1 ? indices.field_1 : (_temp_var_489 == 2 ? indices.field_2 : (_temp_var_489 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_488 == 0 ? indices.field_0 : (_temp_var_488 == 1 ? indices.field_1 : (_temp_var_488 == 2 ? indices.field_2 : (_temp_var_488 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_329_ is already defined
#ifndef _block_k_329__func
#define _block_k_329__func
__device__ int _block_k_329_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_491 = ((({ int _temp_var_492 = ((({ int _temp_var_493 = ((i % 4));
        (_temp_var_493 == 0 ? indices.field_0 : (_temp_var_493 == 1 ? indices.field_1 : (_temp_var_493 == 2 ? indices.field_2 : (_temp_var_493 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_492 == 0 ? indices.field_0 : (_temp_var_492 == 1 ? indices.field_1 : (_temp_var_492 == 2 ? indices.field_2 : (_temp_var_492 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_491 == 0 ? indices.field_0 : (_temp_var_491 == 1 ? indices.field_1 : (_temp_var_491 == 2 ? indices.field_2 : (_temp_var_491 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_331_ is already defined
#ifndef _block_k_331__func
#define _block_k_331__func
__device__ int _block_k_331_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_494 = ((({ int _temp_var_495 = ((({ int _temp_var_496 = ((i % 4));
        (_temp_var_496 == 0 ? indices.field_0 : (_temp_var_496 == 1 ? indices.field_1 : (_temp_var_496 == 2 ? indices.field_2 : (_temp_var_496 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_495 == 0 ? indices.field_0 : (_temp_var_495 == 1 ? indices.field_1 : (_temp_var_495 == 2 ? indices.field_2 : (_temp_var_495 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_494 == 0 ? indices.field_0 : (_temp_var_494 == 1 ? indices.field_1 : (_temp_var_494 == 2 ? indices.field_2 : (_temp_var_494 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_333_ is already defined
#ifndef _block_k_333__func
#define _block_k_333__func
__device__ int _block_k_333_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_497 = ((({ int _temp_var_498 = ((({ int _temp_var_499 = ((i % 4));
        (_temp_var_499 == 0 ? indices.field_0 : (_temp_var_499 == 1 ? indices.field_1 : (_temp_var_499 == 2 ? indices.field_2 : (_temp_var_499 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_498 == 0 ? indices.field_0 : (_temp_var_498 == 1 ? indices.field_1 : (_temp_var_498 == 2 ? indices.field_2 : (_temp_var_498 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_497 == 0 ? indices.field_0 : (_temp_var_497 == 1 ? indices.field_1 : (_temp_var_497 == 2 ? indices.field_2 : (_temp_var_497 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_335_ is already defined
#ifndef _block_k_335__func
#define _block_k_335__func
__device__ int _block_k_335_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_500 = ((({ int _temp_var_501 = ((({ int _temp_var_502 = ((i % 4));
        (_temp_var_502 == 0 ? indices.field_0 : (_temp_var_502 == 1 ? indices.field_1 : (_temp_var_502 == 2 ? indices.field_2 : (_temp_var_502 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_501 == 0 ? indices.field_0 : (_temp_var_501 == 1 ? indices.field_1 : (_temp_var_501 == 2 ? indices.field_2 : (_temp_var_501 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_500 == 0 ? indices.field_0 : (_temp_var_500 == 1 ? indices.field_1 : (_temp_var_500 == 2 ? indices.field_2 : (_temp_var_500 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_337_ is already defined
#ifndef _block_k_337__func
#define _block_k_337__func
__device__ int _block_k_337_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_503 = ((({ int _temp_var_504 = ((({ int _temp_var_505 = ((i % 4));
        (_temp_var_505 == 0 ? indices.field_0 : (_temp_var_505 == 1 ? indices.field_1 : (_temp_var_505 == 2 ? indices.field_2 : (_temp_var_505 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_504 == 0 ? indices.field_0 : (_temp_var_504 == 1 ? indices.field_1 : (_temp_var_504 == 2 ? indices.field_2 : (_temp_var_504 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_503 == 0 ? indices.field_0 : (_temp_var_503 == 1 ? indices.field_1 : (_temp_var_503 == 2 ? indices.field_2 : (_temp_var_503 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_339_ is already defined
#ifndef _block_k_339__func
#define _block_k_339__func
__device__ int _block_k_339_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_506 = ((({ int _temp_var_507 = ((({ int _temp_var_508 = ((i % 4));
        (_temp_var_508 == 0 ? indices.field_0 : (_temp_var_508 == 1 ? indices.field_1 : (_temp_var_508 == 2 ? indices.field_2 : (_temp_var_508 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_507 == 0 ? indices.field_0 : (_temp_var_507 == 1 ? indices.field_1 : (_temp_var_507 == 2 ? indices.field_2 : (_temp_var_507 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_506 == 0 ? indices.field_0 : (_temp_var_506 == 1 ? indices.field_1 : (_temp_var_506 == 2 ? indices.field_2 : (_temp_var_506 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_341_ is already defined
#ifndef _block_k_341__func
#define _block_k_341__func
__device__ int _block_k_341_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_509 = ((({ int _temp_var_510 = ((({ int _temp_var_511 = ((i % 4));
        (_temp_var_511 == 0 ? indices.field_0 : (_temp_var_511 == 1 ? indices.field_1 : (_temp_var_511 == 2 ? indices.field_2 : (_temp_var_511 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_510 == 0 ? indices.field_0 : (_temp_var_510 == 1 ? indices.field_1 : (_temp_var_510 == 2 ? indices.field_2 : (_temp_var_510 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_509 == 0 ? indices.field_0 : (_temp_var_509 == 1 ? indices.field_1 : (_temp_var_509 == 2 ? indices.field_2 : (_temp_var_509 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_343_ is already defined
#ifndef _block_k_343__func
#define _block_k_343__func
__device__ int _block_k_343_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_512 = ((({ int _temp_var_513 = ((({ int _temp_var_514 = ((i % 4));
        (_temp_var_514 == 0 ? indices.field_0 : (_temp_var_514 == 1 ? indices.field_1 : (_temp_var_514 == 2 ? indices.field_2 : (_temp_var_514 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_513 == 0 ? indices.field_0 : (_temp_var_513 == 1 ? indices.field_1 : (_temp_var_513 == 2 ? indices.field_2 : (_temp_var_513 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_512 == 0 ? indices.field_0 : (_temp_var_512 == 1 ? indices.field_1 : (_temp_var_512 == 2 ? indices.field_2 : (_temp_var_512 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_345_ is already defined
#ifndef _block_k_345__func
#define _block_k_345__func
__device__ int _block_k_345_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_515 = ((({ int _temp_var_516 = ((({ int _temp_var_517 = ((i % 4));
        (_temp_var_517 == 0 ? indices.field_0 : (_temp_var_517 == 1 ? indices.field_1 : (_temp_var_517 == 2 ? indices.field_2 : (_temp_var_517 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_516 == 0 ? indices.field_0 : (_temp_var_516 == 1 ? indices.field_1 : (_temp_var_516 == 2 ? indices.field_2 : (_temp_var_516 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_515 == 0 ? indices.field_0 : (_temp_var_515 == 1 ? indices.field_1 : (_temp_var_515 == 2 ? indices.field_2 : (_temp_var_515 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_347_ is already defined
#ifndef _block_k_347__func
#define _block_k_347__func
__device__ int _block_k_347_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_518 = ((({ int _temp_var_519 = ((({ int _temp_var_520 = ((i % 4));
        (_temp_var_520 == 0 ? indices.field_0 : (_temp_var_520 == 1 ? indices.field_1 : (_temp_var_520 == 2 ? indices.field_2 : (_temp_var_520 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_519 == 0 ? indices.field_0 : (_temp_var_519 == 1 ? indices.field_1 : (_temp_var_519 == 2 ? indices.field_2 : (_temp_var_519 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_518 == 0 ? indices.field_0 : (_temp_var_518 == 1 ? indices.field_1 : (_temp_var_518 == 2 ? indices.field_2 : (_temp_var_518 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_349_ is already defined
#ifndef _block_k_349__func
#define _block_k_349__func
__device__ int _block_k_349_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_521 = ((({ int _temp_var_522 = ((({ int _temp_var_523 = ((i % 4));
        (_temp_var_523 == 0 ? indices.field_0 : (_temp_var_523 == 1 ? indices.field_1 : (_temp_var_523 == 2 ? indices.field_2 : (_temp_var_523 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_522 == 0 ? indices.field_0 : (_temp_var_522 == 1 ? indices.field_1 : (_temp_var_522 == 2 ? indices.field_2 : (_temp_var_522 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_521 == 0 ? indices.field_0 : (_temp_var_521 == 1 ? indices.field_1 : (_temp_var_521 == 2 ? indices.field_2 : (_temp_var_521 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_351_ is already defined
#ifndef _block_k_351__func
#define _block_k_351__func
__device__ int _block_k_351_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_524 = ((({ int _temp_var_525 = ((({ int _temp_var_526 = ((i % 4));
        (_temp_var_526 == 0 ? indices.field_0 : (_temp_var_526 == 1 ? indices.field_1 : (_temp_var_526 == 2 ? indices.field_2 : (_temp_var_526 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_525 == 0 ? indices.field_0 : (_temp_var_525 == 1 ? indices.field_1 : (_temp_var_525 == 2 ? indices.field_2 : (_temp_var_525 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_524 == 0 ? indices.field_0 : (_temp_var_524 == 1 ? indices.field_1 : (_temp_var_524 == 2 ? indices.field_2 : (_temp_var_524 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_353_ is already defined
#ifndef _block_k_353__func
#define _block_k_353__func
__device__ int _block_k_353_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_527 = ((({ int _temp_var_528 = ((({ int _temp_var_529 = ((i % 4));
        (_temp_var_529 == 0 ? indices.field_0 : (_temp_var_529 == 1 ? indices.field_1 : (_temp_var_529 == 2 ? indices.field_2 : (_temp_var_529 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_528 == 0 ? indices.field_0 : (_temp_var_528 == 1 ? indices.field_1 : (_temp_var_528 == 2 ? indices.field_2 : (_temp_var_528 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_527 == 0 ? indices.field_0 : (_temp_var_527 == 1 ? indices.field_1 : (_temp_var_527 == 2 ? indices.field_2 : (_temp_var_527 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_355_ is already defined
#ifndef _block_k_355__func
#define _block_k_355__func
__device__ int _block_k_355_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_530 = ((({ int _temp_var_531 = ((({ int _temp_var_532 = ((i % 4));
        (_temp_var_532 == 0 ? indices.field_0 : (_temp_var_532 == 1 ? indices.field_1 : (_temp_var_532 == 2 ? indices.field_2 : (_temp_var_532 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_531 == 0 ? indices.field_0 : (_temp_var_531 == 1 ? indices.field_1 : (_temp_var_531 == 2 ? indices.field_2 : (_temp_var_531 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_530 == 0 ? indices.field_0 : (_temp_var_530 == 1 ? indices.field_1 : (_temp_var_530 == 2 ? indices.field_2 : (_temp_var_530 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_357_ is already defined
#ifndef _block_k_357__func
#define _block_k_357__func
__device__ int _block_k_357_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_533 = ((({ int _temp_var_534 = ((({ int _temp_var_535 = ((i % 4));
        (_temp_var_535 == 0 ? indices.field_0 : (_temp_var_535 == 1 ? indices.field_1 : (_temp_var_535 == 2 ? indices.field_2 : (_temp_var_535 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_534 == 0 ? indices.field_0 : (_temp_var_534 == 1 ? indices.field_1 : (_temp_var_534 == 2 ? indices.field_2 : (_temp_var_534 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_533 == 0 ? indices.field_0 : (_temp_var_533 == 1 ? indices.field_1 : (_temp_var_533 == 2 ? indices.field_2 : (_temp_var_533 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_359_ is already defined
#ifndef _block_k_359__func
#define _block_k_359__func
__device__ int _block_k_359_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_536 = ((({ int _temp_var_537 = ((({ int _temp_var_538 = ((i % 4));
        (_temp_var_538 == 0 ? indices.field_0 : (_temp_var_538 == 1 ? indices.field_1 : (_temp_var_538 == 2 ? indices.field_2 : (_temp_var_538 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_537 == 0 ? indices.field_0 : (_temp_var_537 == 1 ? indices.field_1 : (_temp_var_537 == 2 ? indices.field_2 : (_temp_var_537 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_536 == 0 ? indices.field_0 : (_temp_var_536 == 1 ? indices.field_1 : (_temp_var_536 == 2 ? indices.field_2 : (_temp_var_536 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_361_ is already defined
#ifndef _block_k_361__func
#define _block_k_361__func
__device__ int _block_k_361_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_539 = ((({ int _temp_var_540 = ((({ int _temp_var_541 = ((i % 4));
        (_temp_var_541 == 0 ? indices.field_0 : (_temp_var_541 == 1 ? indices.field_1 : (_temp_var_541 == 2 ? indices.field_2 : (_temp_var_541 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_540 == 0 ? indices.field_0 : (_temp_var_540 == 1 ? indices.field_1 : (_temp_var_540 == 2 ? indices.field_2 : (_temp_var_540 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_539 == 0 ? indices.field_0 : (_temp_var_539 == 1 ? indices.field_1 : (_temp_var_539 == 2 ? indices.field_2 : (_temp_var_539 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_363_ is already defined
#ifndef _block_k_363__func
#define _block_k_363__func
__device__ int _block_k_363_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_542 = ((({ int _temp_var_543 = ((({ int _temp_var_544 = ((i % 4));
        (_temp_var_544 == 0 ? indices.field_0 : (_temp_var_544 == 1 ? indices.field_1 : (_temp_var_544 == 2 ? indices.field_2 : (_temp_var_544 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_543 == 0 ? indices.field_0 : (_temp_var_543 == 1 ? indices.field_1 : (_temp_var_543 == 2 ? indices.field_2 : (_temp_var_543 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_542 == 0 ? indices.field_0 : (_temp_var_542 == 1 ? indices.field_1 : (_temp_var_542 == 2 ? indices.field_2 : (_temp_var_542 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_365_ is already defined
#ifndef _block_k_365__func
#define _block_k_365__func
__device__ int _block_k_365_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_545 = ((({ int _temp_var_546 = ((({ int _temp_var_547 = ((i % 4));
        (_temp_var_547 == 0 ? indices.field_0 : (_temp_var_547 == 1 ? indices.field_1 : (_temp_var_547 == 2 ? indices.field_2 : (_temp_var_547 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_546 == 0 ? indices.field_0 : (_temp_var_546 == 1 ? indices.field_1 : (_temp_var_546 == 2 ? indices.field_2 : (_temp_var_546 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_545 == 0 ? indices.field_0 : (_temp_var_545 == 1 ? indices.field_1 : (_temp_var_545 == 2 ? indices.field_2 : (_temp_var_545 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_367_ is already defined
#ifndef _block_k_367__func
#define _block_k_367__func
__device__ int _block_k_367_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_548 = ((({ int _temp_var_549 = ((({ int _temp_var_550 = ((i % 4));
        (_temp_var_550 == 0 ? indices.field_0 : (_temp_var_550 == 1 ? indices.field_1 : (_temp_var_550 == 2 ? indices.field_2 : (_temp_var_550 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_549 == 0 ? indices.field_0 : (_temp_var_549 == 1 ? indices.field_1 : (_temp_var_549 == 2 ? indices.field_2 : (_temp_var_549 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_548 == 0 ? indices.field_0 : (_temp_var_548 == 1 ? indices.field_1 : (_temp_var_548 == 2 ? indices.field_2 : (_temp_var_548 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_369_ is already defined
#ifndef _block_k_369__func
#define _block_k_369__func
__device__ int _block_k_369_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_551 = ((({ int _temp_var_552 = ((({ int _temp_var_553 = ((i % 4));
        (_temp_var_553 == 0 ? indices.field_0 : (_temp_var_553 == 1 ? indices.field_1 : (_temp_var_553 == 2 ? indices.field_2 : (_temp_var_553 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_552 == 0 ? indices.field_0 : (_temp_var_552 == 1 ? indices.field_1 : (_temp_var_552 == 2 ? indices.field_2 : (_temp_var_552 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_551 == 0 ? indices.field_0 : (_temp_var_551 == 1 ? indices.field_1 : (_temp_var_551 == 2 ? indices.field_2 : (_temp_var_551 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_371_ is already defined
#ifndef _block_k_371__func
#define _block_k_371__func
__device__ int _block_k_371_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_554 = ((({ int _temp_var_555 = ((({ int _temp_var_556 = ((i % 4));
        (_temp_var_556 == 0 ? indices.field_0 : (_temp_var_556 == 1 ? indices.field_1 : (_temp_var_556 == 2 ? indices.field_2 : (_temp_var_556 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_555 == 0 ? indices.field_0 : (_temp_var_555 == 1 ? indices.field_1 : (_temp_var_555 == 2 ? indices.field_2 : (_temp_var_555 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_554 == 0 ? indices.field_0 : (_temp_var_554 == 1 ? indices.field_1 : (_temp_var_554 == 2 ? indices.field_2 : (_temp_var_554 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_373_ is already defined
#ifndef _block_k_373__func
#define _block_k_373__func
__device__ int _block_k_373_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_557 = ((({ int _temp_var_558 = ((({ int _temp_var_559 = ((i % 4));
        (_temp_var_559 == 0 ? indices.field_0 : (_temp_var_559 == 1 ? indices.field_1 : (_temp_var_559 == 2 ? indices.field_2 : (_temp_var_559 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_558 == 0 ? indices.field_0 : (_temp_var_558 == 1 ? indices.field_1 : (_temp_var_558 == 2 ? indices.field_2 : (_temp_var_558 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_557 == 0 ? indices.field_0 : (_temp_var_557 == 1 ? indices.field_1 : (_temp_var_557 == 2 ? indices.field_2 : (_temp_var_557 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_375_ is already defined
#ifndef _block_k_375__func
#define _block_k_375__func
__device__ int _block_k_375_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_560 = ((({ int _temp_var_561 = ((({ int _temp_var_562 = ((i % 4));
        (_temp_var_562 == 0 ? indices.field_0 : (_temp_var_562 == 1 ? indices.field_1 : (_temp_var_562 == 2 ? indices.field_2 : (_temp_var_562 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_561 == 0 ? indices.field_0 : (_temp_var_561 == 1 ? indices.field_1 : (_temp_var_561 == 2 ? indices.field_2 : (_temp_var_561 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_560 == 0 ? indices.field_0 : (_temp_var_560 == 1 ? indices.field_1 : (_temp_var_560 == 2 ? indices.field_2 : (_temp_var_560 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_377_ is already defined
#ifndef _block_k_377__func
#define _block_k_377__func
__device__ int _block_k_377_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_563 = ((({ int _temp_var_564 = ((({ int _temp_var_565 = ((i % 4));
        (_temp_var_565 == 0 ? indices.field_0 : (_temp_var_565 == 1 ? indices.field_1 : (_temp_var_565 == 2 ? indices.field_2 : (_temp_var_565 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_564 == 0 ? indices.field_0 : (_temp_var_564 == 1 ? indices.field_1 : (_temp_var_564 == 2 ? indices.field_2 : (_temp_var_564 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_563 == 0 ? indices.field_0 : (_temp_var_563 == 1 ? indices.field_1 : (_temp_var_563 == 2 ? indices.field_2 : (_temp_var_563 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_379_ is already defined
#ifndef _block_k_379__func
#define _block_k_379__func
__device__ int _block_k_379_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_566 = ((({ int _temp_var_567 = ((({ int _temp_var_568 = ((i % 4));
        (_temp_var_568 == 0 ? indices.field_0 : (_temp_var_568 == 1 ? indices.field_1 : (_temp_var_568 == 2 ? indices.field_2 : (_temp_var_568 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_567 == 0 ? indices.field_0 : (_temp_var_567 == 1 ? indices.field_1 : (_temp_var_567 == 2 ? indices.field_2 : (_temp_var_567 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_566 == 0 ? indices.field_0 : (_temp_var_566 == 1 ? indices.field_1 : (_temp_var_566 == 2 ? indices.field_2 : (_temp_var_566 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_381_ is already defined
#ifndef _block_k_381__func
#define _block_k_381__func
__device__ int _block_k_381_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_569 = ((({ int _temp_var_570 = ((({ int _temp_var_571 = ((i % 4));
        (_temp_var_571 == 0 ? indices.field_0 : (_temp_var_571 == 1 ? indices.field_1 : (_temp_var_571 == 2 ? indices.field_2 : (_temp_var_571 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_570 == 0 ? indices.field_0 : (_temp_var_570 == 1 ? indices.field_1 : (_temp_var_570 == 2 ? indices.field_2 : (_temp_var_570 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_569 == 0 ? indices.field_0 : (_temp_var_569 == 1 ? indices.field_1 : (_temp_var_569 == 2 ? indices.field_2 : (_temp_var_569 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_383_ is already defined
#ifndef _block_k_383__func
#define _block_k_383__func
__device__ int _block_k_383_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_572 = ((({ int _temp_var_573 = ((({ int _temp_var_574 = ((i % 4));
        (_temp_var_574 == 0 ? indices.field_0 : (_temp_var_574 == 1 ? indices.field_1 : (_temp_var_574 == 2 ? indices.field_2 : (_temp_var_574 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_573 == 0 ? indices.field_0 : (_temp_var_573 == 1 ? indices.field_1 : (_temp_var_573 == 2 ? indices.field_2 : (_temp_var_573 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_572 == 0 ? indices.field_0 : (_temp_var_572 == 1 ? indices.field_1 : (_temp_var_572 == 2 ? indices.field_2 : (_temp_var_572 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_385_ is already defined
#ifndef _block_k_385__func
#define _block_k_385__func
__device__ int _block_k_385_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_575 = ((({ int _temp_var_576 = ((({ int _temp_var_577 = ((i % 4));
        (_temp_var_577 == 0 ? indices.field_0 : (_temp_var_577 == 1 ? indices.field_1 : (_temp_var_577 == 2 ? indices.field_2 : (_temp_var_577 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_576 == 0 ? indices.field_0 : (_temp_var_576 == 1 ? indices.field_1 : (_temp_var_576 == 2 ? indices.field_2 : (_temp_var_576 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_575 == 0 ? indices.field_0 : (_temp_var_575 == 1 ? indices.field_1 : (_temp_var_575 == 2 ? indices.field_2 : (_temp_var_575 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_387_ is already defined
#ifndef _block_k_387__func
#define _block_k_387__func
__device__ int _block_k_387_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_578 = ((({ int _temp_var_579 = ((({ int _temp_var_580 = ((i % 4));
        (_temp_var_580 == 0 ? indices.field_0 : (_temp_var_580 == 1 ? indices.field_1 : (_temp_var_580 == 2 ? indices.field_2 : (_temp_var_580 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_579 == 0 ? indices.field_0 : (_temp_var_579 == 1 ? indices.field_1 : (_temp_var_579 == 2 ? indices.field_2 : (_temp_var_579 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_578 == 0 ? indices.field_0 : (_temp_var_578 == 1 ? indices.field_1 : (_temp_var_578 == 2 ? indices.field_2 : (_temp_var_578 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_389_ is already defined
#ifndef _block_k_389__func
#define _block_k_389__func
__device__ int _block_k_389_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_581 = ((({ int _temp_var_582 = ((({ int _temp_var_583 = ((i % 4));
        (_temp_var_583 == 0 ? indices.field_0 : (_temp_var_583 == 1 ? indices.field_1 : (_temp_var_583 == 2 ? indices.field_2 : (_temp_var_583 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_582 == 0 ? indices.field_0 : (_temp_var_582 == 1 ? indices.field_1 : (_temp_var_582 == 2 ? indices.field_2 : (_temp_var_582 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_581 == 0 ? indices.field_0 : (_temp_var_581 == 1 ? indices.field_1 : (_temp_var_581 == 2 ? indices.field_2 : (_temp_var_581 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_391_ is already defined
#ifndef _block_k_391__func
#define _block_k_391__func
__device__ int _block_k_391_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_584 = ((({ int _temp_var_585 = ((({ int _temp_var_586 = ((i % 4));
        (_temp_var_586 == 0 ? indices.field_0 : (_temp_var_586 == 1 ? indices.field_1 : (_temp_var_586 == 2 ? indices.field_2 : (_temp_var_586 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_585 == 0 ? indices.field_0 : (_temp_var_585 == 1 ? indices.field_1 : (_temp_var_585 == 2 ? indices.field_2 : (_temp_var_585 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_584 == 0 ? indices.field_0 : (_temp_var_584 == 1 ? indices.field_1 : (_temp_var_584 == 2 ? indices.field_2 : (_temp_var_584 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_393_ is already defined
#ifndef _block_k_393__func
#define _block_k_393__func
__device__ int _block_k_393_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_587 = ((({ int _temp_var_588 = ((({ int _temp_var_589 = ((i % 4));
        (_temp_var_589 == 0 ? indices.field_0 : (_temp_var_589 == 1 ? indices.field_1 : (_temp_var_589 == 2 ? indices.field_2 : (_temp_var_589 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_588 == 0 ? indices.field_0 : (_temp_var_588 == 1 ? indices.field_1 : (_temp_var_588 == 2 ? indices.field_2 : (_temp_var_588 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_587 == 0 ? indices.field_0 : (_temp_var_587 == 1 ? indices.field_1 : (_temp_var_587 == 2 ? indices.field_2 : (_temp_var_587 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_395_ is already defined
#ifndef _block_k_395__func
#define _block_k_395__func
__device__ int _block_k_395_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_590 = ((({ int _temp_var_591 = ((({ int _temp_var_592 = ((i % 4));
        (_temp_var_592 == 0 ? indices.field_0 : (_temp_var_592 == 1 ? indices.field_1 : (_temp_var_592 == 2 ? indices.field_2 : (_temp_var_592 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_591 == 0 ? indices.field_0 : (_temp_var_591 == 1 ? indices.field_1 : (_temp_var_591 == 2 ? indices.field_2 : (_temp_var_591 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_590 == 0 ? indices.field_0 : (_temp_var_590 == 1 ? indices.field_1 : (_temp_var_590 == 2 ? indices.field_2 : (_temp_var_590 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_397_ is already defined
#ifndef _block_k_397__func
#define _block_k_397__func
__device__ int _block_k_397_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_593 = ((({ int _temp_var_594 = ((({ int _temp_var_595 = ((i % 4));
        (_temp_var_595 == 0 ? indices.field_0 : (_temp_var_595 == 1 ? indices.field_1 : (_temp_var_595 == 2 ? indices.field_2 : (_temp_var_595 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_594 == 0 ? indices.field_0 : (_temp_var_594 == 1 ? indices.field_1 : (_temp_var_594 == 2 ? indices.field_2 : (_temp_var_594 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_593 == 0 ? indices.field_0 : (_temp_var_593 == 1 ? indices.field_1 : (_temp_var_593 == 2 ? indices.field_2 : (_temp_var_593 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_399_ is already defined
#ifndef _block_k_399__func
#define _block_k_399__func
__device__ int _block_k_399_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_596 = ((({ int _temp_var_597 = ((({ int _temp_var_598 = ((i % 4));
        (_temp_var_598 == 0 ? indices.field_0 : (_temp_var_598 == 1 ? indices.field_1 : (_temp_var_598 == 2 ? indices.field_2 : (_temp_var_598 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_597 == 0 ? indices.field_0 : (_temp_var_597 == 1 ? indices.field_1 : (_temp_var_597 == 2 ? indices.field_2 : (_temp_var_597 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_596 == 0 ? indices.field_0 : (_temp_var_596 == 1 ? indices.field_1 : (_temp_var_596 == 2 ? indices.field_2 : (_temp_var_596 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_401_ is already defined
#ifndef _block_k_401__func
#define _block_k_401__func
__device__ int _block_k_401_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_599 = ((({ int _temp_var_600 = ((({ int _temp_var_601 = ((i % 4));
        (_temp_var_601 == 0 ? indices.field_0 : (_temp_var_601 == 1 ? indices.field_1 : (_temp_var_601 == 2 ? indices.field_2 : (_temp_var_601 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_600 == 0 ? indices.field_0 : (_temp_var_600 == 1 ? indices.field_1 : (_temp_var_600 == 2 ? indices.field_2 : (_temp_var_600 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_599 == 0 ? indices.field_0 : (_temp_var_599 == 1 ? indices.field_1 : (_temp_var_599 == 2 ? indices.field_2 : (_temp_var_599 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_403_ is already defined
#ifndef _block_k_403__func
#define _block_k_403__func
__device__ int _block_k_403_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_602 = ((({ int _temp_var_603 = ((({ int _temp_var_604 = ((i % 4));
        (_temp_var_604 == 0 ? indices.field_0 : (_temp_var_604 == 1 ? indices.field_1 : (_temp_var_604 == 2 ? indices.field_2 : (_temp_var_604 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_603 == 0 ? indices.field_0 : (_temp_var_603 == 1 ? indices.field_1 : (_temp_var_603 == 2 ? indices.field_2 : (_temp_var_603 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_602 == 0 ? indices.field_0 : (_temp_var_602 == 1 ? indices.field_1 : (_temp_var_602 == 2 ? indices.field_2 : (_temp_var_602 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_405_ is already defined
#ifndef _block_k_405__func
#define _block_k_405__func
__device__ int _block_k_405_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_605 = ((({ int _temp_var_606 = ((({ int _temp_var_607 = ((i % 4));
        (_temp_var_607 == 0 ? indices.field_0 : (_temp_var_607 == 1 ? indices.field_1 : (_temp_var_607 == 2 ? indices.field_2 : (_temp_var_607 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_606 == 0 ? indices.field_0 : (_temp_var_606 == 1 ? indices.field_1 : (_temp_var_606 == 2 ? indices.field_2 : (_temp_var_606 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_605 == 0 ? indices.field_0 : (_temp_var_605 == 1 ? indices.field_1 : (_temp_var_605 == 2 ? indices.field_2 : (_temp_var_605 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_407_ is already defined
#ifndef _block_k_407__func
#define _block_k_407__func
__device__ int _block_k_407_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_608 = ((({ int _temp_var_609 = ((({ int _temp_var_610 = ((i % 4));
        (_temp_var_610 == 0 ? indices.field_0 : (_temp_var_610 == 1 ? indices.field_1 : (_temp_var_610 == 2 ? indices.field_2 : (_temp_var_610 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_609 == 0 ? indices.field_0 : (_temp_var_609 == 1 ? indices.field_1 : (_temp_var_609 == 2 ? indices.field_2 : (_temp_var_609 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_608 == 0 ? indices.field_0 : (_temp_var_608 == 1 ? indices.field_1 : (_temp_var_608 == 2 ? indices.field_2 : (_temp_var_608 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_409_ is already defined
#ifndef _block_k_409__func
#define _block_k_409__func
__device__ int _block_k_409_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_611 = ((({ int _temp_var_612 = ((({ int _temp_var_613 = ((i % 4));
        (_temp_var_613 == 0 ? indices.field_0 : (_temp_var_613 == 1 ? indices.field_1 : (_temp_var_613 == 2 ? indices.field_2 : (_temp_var_613 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_612 == 0 ? indices.field_0 : (_temp_var_612 == 1 ? indices.field_1 : (_temp_var_612 == 2 ? indices.field_2 : (_temp_var_612 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_611 == 0 ? indices.field_0 : (_temp_var_611 == 1 ? indices.field_1 : (_temp_var_611 == 2 ? indices.field_2 : (_temp_var_611 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_411_ is already defined
#ifndef _block_k_411__func
#define _block_k_411__func
__device__ int _block_k_411_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_614 = ((({ int _temp_var_615 = ((({ int _temp_var_616 = ((i % 4));
        (_temp_var_616 == 0 ? indices.field_0 : (_temp_var_616 == 1 ? indices.field_1 : (_temp_var_616 == 2 ? indices.field_2 : (_temp_var_616 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_615 == 0 ? indices.field_0 : (_temp_var_615 == 1 ? indices.field_1 : (_temp_var_615 == 2 ? indices.field_2 : (_temp_var_615 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_614 == 0 ? indices.field_0 : (_temp_var_614 == 1 ? indices.field_1 : (_temp_var_614 == 2 ? indices.field_2 : (_temp_var_614 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_413_ is already defined
#ifndef _block_k_413__func
#define _block_k_413__func
__device__ int _block_k_413_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_617 = ((({ int _temp_var_618 = ((({ int _temp_var_619 = ((i % 4));
        (_temp_var_619 == 0 ? indices.field_0 : (_temp_var_619 == 1 ? indices.field_1 : (_temp_var_619 == 2 ? indices.field_2 : (_temp_var_619 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_618 == 0 ? indices.field_0 : (_temp_var_618 == 1 ? indices.field_1 : (_temp_var_618 == 2 ? indices.field_2 : (_temp_var_618 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_617 == 0 ? indices.field_0 : (_temp_var_617 == 1 ? indices.field_1 : (_temp_var_617 == 2 ? indices.field_2 : (_temp_var_617 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_415_ is already defined
#ifndef _block_k_415__func
#define _block_k_415__func
__device__ int _block_k_415_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_620 = ((({ int _temp_var_621 = ((({ int _temp_var_622 = ((i % 4));
        (_temp_var_622 == 0 ? indices.field_0 : (_temp_var_622 == 1 ? indices.field_1 : (_temp_var_622 == 2 ? indices.field_2 : (_temp_var_622 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_621 == 0 ? indices.field_0 : (_temp_var_621 == 1 ? indices.field_1 : (_temp_var_621 == 2 ? indices.field_2 : (_temp_var_621 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_620 == 0 ? indices.field_0 : (_temp_var_620 == 1 ? indices.field_1 : (_temp_var_620 == 2 ? indices.field_2 : (_temp_var_620 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_417_ is already defined
#ifndef _block_k_417__func
#define _block_k_417__func
__device__ int _block_k_417_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_623 = ((({ int _temp_var_624 = ((({ int _temp_var_625 = ((i % 4));
        (_temp_var_625 == 0 ? indices.field_0 : (_temp_var_625 == 1 ? indices.field_1 : (_temp_var_625 == 2 ? indices.field_2 : (_temp_var_625 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_624 == 0 ? indices.field_0 : (_temp_var_624 == 1 ? indices.field_1 : (_temp_var_624 == 2 ? indices.field_2 : (_temp_var_624 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_623 == 0 ? indices.field_0 : (_temp_var_623 == 1 ? indices.field_1 : (_temp_var_623 == 2 ? indices.field_2 : (_temp_var_623 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_419_ is already defined
#ifndef _block_k_419__func
#define _block_k_419__func
__device__ int _block_k_419_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_626 = ((({ int _temp_var_627 = ((({ int _temp_var_628 = ((i % 4));
        (_temp_var_628 == 0 ? indices.field_0 : (_temp_var_628 == 1 ? indices.field_1 : (_temp_var_628 == 2 ? indices.field_2 : (_temp_var_628 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_627 == 0 ? indices.field_0 : (_temp_var_627 == 1 ? indices.field_1 : (_temp_var_627 == 2 ? indices.field_2 : (_temp_var_627 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_626 == 0 ? indices.field_0 : (_temp_var_626 == 1 ? indices.field_1 : (_temp_var_626 == 2 ? indices.field_2 : (_temp_var_626 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_421_ is already defined
#ifndef _block_k_421__func
#define _block_k_421__func
__device__ int _block_k_421_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_629 = ((({ int _temp_var_630 = ((({ int _temp_var_631 = ((i % 4));
        (_temp_var_631 == 0 ? indices.field_0 : (_temp_var_631 == 1 ? indices.field_1 : (_temp_var_631 == 2 ? indices.field_2 : (_temp_var_631 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_630 == 0 ? indices.field_0 : (_temp_var_630 == 1 ? indices.field_1 : (_temp_var_630 == 2 ? indices.field_2 : (_temp_var_630 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_629 == 0 ? indices.field_0 : (_temp_var_629 == 1 ? indices.field_1 : (_temp_var_629 == 2 ? indices.field_2 : (_temp_var_629 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_423_ is already defined
#ifndef _block_k_423__func
#define _block_k_423__func
__device__ int _block_k_423_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_632 = ((({ int _temp_var_633 = ((({ int _temp_var_634 = ((i % 4));
        (_temp_var_634 == 0 ? indices.field_0 : (_temp_var_634 == 1 ? indices.field_1 : (_temp_var_634 == 2 ? indices.field_2 : (_temp_var_634 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_633 == 0 ? indices.field_0 : (_temp_var_633 == 1 ? indices.field_1 : (_temp_var_633 == 2 ? indices.field_2 : (_temp_var_633 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_632 == 0 ? indices.field_0 : (_temp_var_632 == 1 ? indices.field_1 : (_temp_var_632 == 2 ? indices.field_2 : (_temp_var_632 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_425_ is already defined
#ifndef _block_k_425__func
#define _block_k_425__func
__device__ int _block_k_425_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_635 = ((({ int _temp_var_636 = ((({ int _temp_var_637 = ((i % 4));
        (_temp_var_637 == 0 ? indices.field_0 : (_temp_var_637 == 1 ? indices.field_1 : (_temp_var_637 == 2 ? indices.field_2 : (_temp_var_637 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_636 == 0 ? indices.field_0 : (_temp_var_636 == 1 ? indices.field_1 : (_temp_var_636 == 2 ? indices.field_2 : (_temp_var_636 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_635 == 0 ? indices.field_0 : (_temp_var_635 == 1 ? indices.field_1 : (_temp_var_635 == 2 ? indices.field_2 : (_temp_var_635 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_427_ is already defined
#ifndef _block_k_427__func
#define _block_k_427__func
__device__ int _block_k_427_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_638 = ((({ int _temp_var_639 = ((({ int _temp_var_640 = ((i % 4));
        (_temp_var_640 == 0 ? indices.field_0 : (_temp_var_640 == 1 ? indices.field_1 : (_temp_var_640 == 2 ? indices.field_2 : (_temp_var_640 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_639 == 0 ? indices.field_0 : (_temp_var_639 == 1 ? indices.field_1 : (_temp_var_639 == 2 ? indices.field_2 : (_temp_var_639 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_638 == 0 ? indices.field_0 : (_temp_var_638 == 1 ? indices.field_1 : (_temp_var_638 == 2 ? indices.field_2 : (_temp_var_638 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_429_ is already defined
#ifndef _block_k_429__func
#define _block_k_429__func
__device__ int _block_k_429_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_641 = ((({ int _temp_var_642 = ((({ int _temp_var_643 = ((i % 4));
        (_temp_var_643 == 0 ? indices.field_0 : (_temp_var_643 == 1 ? indices.field_1 : (_temp_var_643 == 2 ? indices.field_2 : (_temp_var_643 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_642 == 0 ? indices.field_0 : (_temp_var_642 == 1 ? indices.field_1 : (_temp_var_642 == 2 ? indices.field_2 : (_temp_var_642 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_641 == 0 ? indices.field_0 : (_temp_var_641 == 1 ? indices.field_1 : (_temp_var_641 == 2 ? indices.field_2 : (_temp_var_641 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_431_ is already defined
#ifndef _block_k_431__func
#define _block_k_431__func
__device__ int _block_k_431_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_644 = ((({ int _temp_var_645 = ((({ int _temp_var_646 = ((i % 4));
        (_temp_var_646 == 0 ? indices.field_0 : (_temp_var_646 == 1 ? indices.field_1 : (_temp_var_646 == 2 ? indices.field_2 : (_temp_var_646 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_645 == 0 ? indices.field_0 : (_temp_var_645 == 1 ? indices.field_1 : (_temp_var_645 == 2 ? indices.field_2 : (_temp_var_645 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_644 == 0 ? indices.field_0 : (_temp_var_644 == 1 ? indices.field_1 : (_temp_var_644 == 2 ? indices.field_2 : (_temp_var_644 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_433_ is already defined
#ifndef _block_k_433__func
#define _block_k_433__func
__device__ int _block_k_433_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_647 = ((({ int _temp_var_648 = ((({ int _temp_var_649 = ((i % 4));
        (_temp_var_649 == 0 ? indices.field_0 : (_temp_var_649 == 1 ? indices.field_1 : (_temp_var_649 == 2 ? indices.field_2 : (_temp_var_649 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_648 == 0 ? indices.field_0 : (_temp_var_648 == 1 ? indices.field_1 : (_temp_var_648 == 2 ? indices.field_2 : (_temp_var_648 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_647 == 0 ? indices.field_0 : (_temp_var_647 == 1 ? indices.field_1 : (_temp_var_647 == 2 ? indices.field_2 : (_temp_var_647 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_435_ is already defined
#ifndef _block_k_435__func
#define _block_k_435__func
__device__ int _block_k_435_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_650 = ((({ int _temp_var_651 = ((({ int _temp_var_652 = ((i % 4));
        (_temp_var_652 == 0 ? indices.field_0 : (_temp_var_652 == 1 ? indices.field_1 : (_temp_var_652 == 2 ? indices.field_2 : (_temp_var_652 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_651 == 0 ? indices.field_0 : (_temp_var_651 == 1 ? indices.field_1 : (_temp_var_651 == 2 ? indices.field_2 : (_temp_var_651 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_650 == 0 ? indices.field_0 : (_temp_var_650 == 1 ? indices.field_1 : (_temp_var_650 == 2 ? indices.field_2 : (_temp_var_650 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_437_ is already defined
#ifndef _block_k_437__func
#define _block_k_437__func
__device__ int _block_k_437_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_653 = ((({ int _temp_var_654 = ((({ int _temp_var_655 = ((i % 4));
        (_temp_var_655 == 0 ? indices.field_0 : (_temp_var_655 == 1 ? indices.field_1 : (_temp_var_655 == 2 ? indices.field_2 : (_temp_var_655 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_654 == 0 ? indices.field_0 : (_temp_var_654 == 1 ? indices.field_1 : (_temp_var_654 == 2 ? indices.field_2 : (_temp_var_654 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_653 == 0 ? indices.field_0 : (_temp_var_653 == 1 ? indices.field_1 : (_temp_var_653 == 2 ? indices.field_2 : (_temp_var_653 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_439_ is already defined
#ifndef _block_k_439__func
#define _block_k_439__func
__device__ int _block_k_439_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_656 = ((({ int _temp_var_657 = ((({ int _temp_var_658 = ((i % 4));
        (_temp_var_658 == 0 ? indices.field_0 : (_temp_var_658 == 1 ? indices.field_1 : (_temp_var_658 == 2 ? indices.field_2 : (_temp_var_658 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_657 == 0 ? indices.field_0 : (_temp_var_657 == 1 ? indices.field_1 : (_temp_var_657 == 2 ? indices.field_2 : (_temp_var_657 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_656 == 0 ? indices.field_0 : (_temp_var_656 == 1 ? indices.field_1 : (_temp_var_656 == 2 ? indices.field_2 : (_temp_var_656 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_441_ is already defined
#ifndef _block_k_441__func
#define _block_k_441__func
__device__ int _block_k_441_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_659 = ((({ int _temp_var_660 = ((({ int _temp_var_661 = ((i % 4));
        (_temp_var_661 == 0 ? indices.field_0 : (_temp_var_661 == 1 ? indices.field_1 : (_temp_var_661 == 2 ? indices.field_2 : (_temp_var_661 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_660 == 0 ? indices.field_0 : (_temp_var_660 == 1 ? indices.field_1 : (_temp_var_660 == 2 ? indices.field_2 : (_temp_var_660 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_659 == 0 ? indices.field_0 : (_temp_var_659 == 1 ? indices.field_1 : (_temp_var_659 == 2 ? indices.field_2 : (_temp_var_659 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_443_ is already defined
#ifndef _block_k_443__func
#define _block_k_443__func
__device__ int _block_k_443_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_662 = ((({ int _temp_var_663 = ((({ int _temp_var_664 = ((i % 4));
        (_temp_var_664 == 0 ? indices.field_0 : (_temp_var_664 == 1 ? indices.field_1 : (_temp_var_664 == 2 ? indices.field_2 : (_temp_var_664 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_663 == 0 ? indices.field_0 : (_temp_var_663 == 1 ? indices.field_1 : (_temp_var_663 == 2 ? indices.field_2 : (_temp_var_663 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_662 == 0 ? indices.field_0 : (_temp_var_662 == 1 ? indices.field_1 : (_temp_var_662 == 2 ? indices.field_2 : (_temp_var_662 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_445_ is already defined
#ifndef _block_k_445__func
#define _block_k_445__func
__device__ int _block_k_445_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_665 = ((({ int _temp_var_666 = ((({ int _temp_var_667 = ((i % 4));
        (_temp_var_667 == 0 ? indices.field_0 : (_temp_var_667 == 1 ? indices.field_1 : (_temp_var_667 == 2 ? indices.field_2 : (_temp_var_667 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_666 == 0 ? indices.field_0 : (_temp_var_666 == 1 ? indices.field_1 : (_temp_var_666 == 2 ? indices.field_2 : (_temp_var_666 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_665 == 0 ? indices.field_0 : (_temp_var_665 == 1 ? indices.field_1 : (_temp_var_665 == 2 ? indices.field_2 : (_temp_var_665 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_447_ is already defined
#ifndef _block_k_447__func
#define _block_k_447__func
__device__ int _block_k_447_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_668 = ((({ int _temp_var_669 = ((({ int _temp_var_670 = ((i % 4));
        (_temp_var_670 == 0 ? indices.field_0 : (_temp_var_670 == 1 ? indices.field_1 : (_temp_var_670 == 2 ? indices.field_2 : (_temp_var_670 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_669 == 0 ? indices.field_0 : (_temp_var_669 == 1 ? indices.field_1 : (_temp_var_669 == 2 ? indices.field_2 : (_temp_var_669 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_668 == 0 ? indices.field_0 : (_temp_var_668 == 1 ? indices.field_1 : (_temp_var_668 == 2 ? indices.field_2 : (_temp_var_668 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_449_ is already defined
#ifndef _block_k_449__func
#define _block_k_449__func
__device__ int _block_k_449_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_671 = ((({ int _temp_var_672 = ((({ int _temp_var_673 = ((i % 4));
        (_temp_var_673 == 0 ? indices.field_0 : (_temp_var_673 == 1 ? indices.field_1 : (_temp_var_673 == 2 ? indices.field_2 : (_temp_var_673 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_672 == 0 ? indices.field_0 : (_temp_var_672 == 1 ? indices.field_1 : (_temp_var_672 == 2 ? indices.field_2 : (_temp_var_672 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_671 == 0 ? indices.field_0 : (_temp_var_671 == 1 ? indices.field_1 : (_temp_var_671 == 2 ? indices.field_2 : (_temp_var_671 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_451_ is already defined
#ifndef _block_k_451__func
#define _block_k_451__func
__device__ int _block_k_451_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_674 = ((({ int _temp_var_675 = ((({ int _temp_var_676 = ((i % 4));
        (_temp_var_676 == 0 ? indices.field_0 : (_temp_var_676 == 1 ? indices.field_1 : (_temp_var_676 == 2 ? indices.field_2 : (_temp_var_676 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_675 == 0 ? indices.field_0 : (_temp_var_675 == 1 ? indices.field_1 : (_temp_var_675 == 2 ? indices.field_2 : (_temp_var_675 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_674 == 0 ? indices.field_0 : (_temp_var_674 == 1 ? indices.field_1 : (_temp_var_674 == 2 ? indices.field_2 : (_temp_var_674 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_453_ is already defined
#ifndef _block_k_453__func
#define _block_k_453__func
__device__ int _block_k_453_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_677 = ((({ int _temp_var_678 = ((({ int _temp_var_679 = ((i % 4));
        (_temp_var_679 == 0 ? indices.field_0 : (_temp_var_679 == 1 ? indices.field_1 : (_temp_var_679 == 2 ? indices.field_2 : (_temp_var_679 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_678 == 0 ? indices.field_0 : (_temp_var_678 == 1 ? indices.field_1 : (_temp_var_678 == 2 ? indices.field_2 : (_temp_var_678 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_677 == 0 ? indices.field_0 : (_temp_var_677 == 1 ? indices.field_1 : (_temp_var_677 == 2 ? indices.field_2 : (_temp_var_677 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_455_ is already defined
#ifndef _block_k_455__func
#define _block_k_455__func
__device__ int _block_k_455_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_680 = ((({ int _temp_var_681 = ((({ int _temp_var_682 = ((i % 4));
        (_temp_var_682 == 0 ? indices.field_0 : (_temp_var_682 == 1 ? indices.field_1 : (_temp_var_682 == 2 ? indices.field_2 : (_temp_var_682 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_681 == 0 ? indices.field_0 : (_temp_var_681 == 1 ? indices.field_1 : (_temp_var_681 == 2 ? indices.field_2 : (_temp_var_681 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_680 == 0 ? indices.field_0 : (_temp_var_680 == 1 ? indices.field_1 : (_temp_var_680 == 2 ? indices.field_2 : (_temp_var_680 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_457_ is already defined
#ifndef _block_k_457__func
#define _block_k_457__func
__device__ int _block_k_457_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_683 = ((({ int _temp_var_684 = ((({ int _temp_var_685 = ((i % 4));
        (_temp_var_685 == 0 ? indices.field_0 : (_temp_var_685 == 1 ? indices.field_1 : (_temp_var_685 == 2 ? indices.field_2 : (_temp_var_685 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_684 == 0 ? indices.field_0 : (_temp_var_684 == 1 ? indices.field_1 : (_temp_var_684 == 2 ? indices.field_2 : (_temp_var_684 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_683 == 0 ? indices.field_0 : (_temp_var_683 == 1 ? indices.field_1 : (_temp_var_683 == 2 ? indices.field_2 : (_temp_var_683 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_459_ is already defined
#ifndef _block_k_459__func
#define _block_k_459__func
__device__ int _block_k_459_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_686 = ((({ int _temp_var_687 = ((({ int _temp_var_688 = ((i % 4));
        (_temp_var_688 == 0 ? indices.field_0 : (_temp_var_688 == 1 ? indices.field_1 : (_temp_var_688 == 2 ? indices.field_2 : (_temp_var_688 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_687 == 0 ? indices.field_0 : (_temp_var_687 == 1 ? indices.field_1 : (_temp_var_687 == 2 ? indices.field_2 : (_temp_var_687 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_686 == 0 ? indices.field_0 : (_temp_var_686 == 1 ? indices.field_1 : (_temp_var_686 == 2 ? indices.field_2 : (_temp_var_686 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_461_ is already defined
#ifndef _block_k_461__func
#define _block_k_461__func
__device__ int _block_k_461_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_689 = ((({ int _temp_var_690 = ((({ int _temp_var_691 = ((i % 4));
        (_temp_var_691 == 0 ? indices.field_0 : (_temp_var_691 == 1 ? indices.field_1 : (_temp_var_691 == 2 ? indices.field_2 : (_temp_var_691 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_690 == 0 ? indices.field_0 : (_temp_var_690 == 1 ? indices.field_1 : (_temp_var_690 == 2 ? indices.field_2 : (_temp_var_690 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_689 == 0 ? indices.field_0 : (_temp_var_689 == 1 ? indices.field_1 : (_temp_var_689 == 2 ? indices.field_2 : (_temp_var_689 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_463_ is already defined
#ifndef _block_k_463__func
#define _block_k_463__func
__device__ int _block_k_463_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_692 = ((({ int _temp_var_693 = ((({ int _temp_var_694 = ((i % 4));
        (_temp_var_694 == 0 ? indices.field_0 : (_temp_var_694 == 1 ? indices.field_1 : (_temp_var_694 == 2 ? indices.field_2 : (_temp_var_694 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_693 == 0 ? indices.field_0 : (_temp_var_693 == 1 ? indices.field_1 : (_temp_var_693 == 2 ? indices.field_2 : (_temp_var_693 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_692 == 0 ? indices.field_0 : (_temp_var_692 == 1 ? indices.field_1 : (_temp_var_692 == 2 ? indices.field_2 : (_temp_var_692 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_465_ is already defined
#ifndef _block_k_465__func
#define _block_k_465__func
__device__ int _block_k_465_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_695 = ((({ int _temp_var_696 = ((({ int _temp_var_697 = ((i % 4));
        (_temp_var_697 == 0 ? indices.field_0 : (_temp_var_697 == 1 ? indices.field_1 : (_temp_var_697 == 2 ? indices.field_2 : (_temp_var_697 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_696 == 0 ? indices.field_0 : (_temp_var_696 == 1 ? indices.field_1 : (_temp_var_696 == 2 ? indices.field_2 : (_temp_var_696 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_695 == 0 ? indices.field_0 : (_temp_var_695 == 1 ? indices.field_1 : (_temp_var_695 == 2 ? indices.field_2 : (_temp_var_695 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_467_ is already defined
#ifndef _block_k_467__func
#define _block_k_467__func
__device__ int _block_k_467_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_698 = ((({ int _temp_var_699 = ((({ int _temp_var_700 = ((i % 4));
        (_temp_var_700 == 0 ? indices.field_0 : (_temp_var_700 == 1 ? indices.field_1 : (_temp_var_700 == 2 ? indices.field_2 : (_temp_var_700 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_699 == 0 ? indices.field_0 : (_temp_var_699 == 1 ? indices.field_1 : (_temp_var_699 == 2 ? indices.field_2 : (_temp_var_699 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_698 == 0 ? indices.field_0 : (_temp_var_698 == 1 ? indices.field_1 : (_temp_var_698 == 2 ? indices.field_2 : (_temp_var_698 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_469_ is already defined
#ifndef _block_k_469__func
#define _block_k_469__func
__device__ int _block_k_469_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_701 = ((({ int _temp_var_702 = ((({ int _temp_var_703 = ((i % 4));
        (_temp_var_703 == 0 ? indices.field_0 : (_temp_var_703 == 1 ? indices.field_1 : (_temp_var_703 == 2 ? indices.field_2 : (_temp_var_703 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_702 == 0 ? indices.field_0 : (_temp_var_702 == 1 ? indices.field_1 : (_temp_var_702 == 2 ? indices.field_2 : (_temp_var_702 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_701 == 0 ? indices.field_0 : (_temp_var_701 == 1 ? indices.field_1 : (_temp_var_701 == 2 ? indices.field_2 : (_temp_var_701 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_471_ is already defined
#ifndef _block_k_471__func
#define _block_k_471__func
__device__ int _block_k_471_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_704 = ((({ int _temp_var_705 = ((({ int _temp_var_706 = ((i % 4));
        (_temp_var_706 == 0 ? indices.field_0 : (_temp_var_706 == 1 ? indices.field_1 : (_temp_var_706 == 2 ? indices.field_2 : (_temp_var_706 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_705 == 0 ? indices.field_0 : (_temp_var_705 == 1 ? indices.field_1 : (_temp_var_705 == 2 ? indices.field_2 : (_temp_var_705 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_704 == 0 ? indices.field_0 : (_temp_var_704 == 1 ? indices.field_1 : (_temp_var_704 == 2 ? indices.field_2 : (_temp_var_704 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_473_ is already defined
#ifndef _block_k_473__func
#define _block_k_473__func
__device__ int _block_k_473_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_707 = ((({ int _temp_var_708 = ((({ int _temp_var_709 = ((i % 4));
        (_temp_var_709 == 0 ? indices.field_0 : (_temp_var_709 == 1 ? indices.field_1 : (_temp_var_709 == 2 ? indices.field_2 : (_temp_var_709 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_708 == 0 ? indices.field_0 : (_temp_var_708 == 1 ? indices.field_1 : (_temp_var_708 == 2 ? indices.field_2 : (_temp_var_708 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_707 == 0 ? indices.field_0 : (_temp_var_707 == 1 ? indices.field_1 : (_temp_var_707 == 2 ? indices.field_2 : (_temp_var_707 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_475_ is already defined
#ifndef _block_k_475__func
#define _block_k_475__func
__device__ int _block_k_475_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_710 = ((({ int _temp_var_711 = ((({ int _temp_var_712 = ((i % 4));
        (_temp_var_712 == 0 ? indices.field_0 : (_temp_var_712 == 1 ? indices.field_1 : (_temp_var_712 == 2 ? indices.field_2 : (_temp_var_712 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_711 == 0 ? indices.field_0 : (_temp_var_711 == 1 ? indices.field_1 : (_temp_var_711 == 2 ? indices.field_2 : (_temp_var_711 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_710 == 0 ? indices.field_0 : (_temp_var_710 == 1 ? indices.field_1 : (_temp_var_710 == 2 ? indices.field_2 : (_temp_var_710 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_477_ is already defined
#ifndef _block_k_477__func
#define _block_k_477__func
__device__ int _block_k_477_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_713 = ((({ int _temp_var_714 = ((({ int _temp_var_715 = ((i % 4));
        (_temp_var_715 == 0 ? indices.field_0 : (_temp_var_715 == 1 ? indices.field_1 : (_temp_var_715 == 2 ? indices.field_2 : (_temp_var_715 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_714 == 0 ? indices.field_0 : (_temp_var_714 == 1 ? indices.field_1 : (_temp_var_714 == 2 ? indices.field_2 : (_temp_var_714 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_713 == 0 ? indices.field_0 : (_temp_var_713 == 1 ? indices.field_1 : (_temp_var_713 == 2 ? indices.field_2 : (_temp_var_713 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_479_ is already defined
#ifndef _block_k_479__func
#define _block_k_479__func
__device__ int _block_k_479_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_716 = ((({ int _temp_var_717 = ((({ int _temp_var_718 = ((i % 4));
        (_temp_var_718 == 0 ? indices.field_0 : (_temp_var_718 == 1 ? indices.field_1 : (_temp_var_718 == 2 ? indices.field_2 : (_temp_var_718 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_717 == 0 ? indices.field_0 : (_temp_var_717 == 1 ? indices.field_1 : (_temp_var_717 == 2 ? indices.field_2 : (_temp_var_717 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_716 == 0 ? indices.field_0 : (_temp_var_716 == 1 ? indices.field_1 : (_temp_var_716 == 2 ? indices.field_2 : (_temp_var_716 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_481_ is already defined
#ifndef _block_k_481__func
#define _block_k_481__func
__device__ int _block_k_481_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_719 = ((({ int _temp_var_720 = ((({ int _temp_var_721 = ((i % 4));
        (_temp_var_721 == 0 ? indices.field_0 : (_temp_var_721 == 1 ? indices.field_1 : (_temp_var_721 == 2 ? indices.field_2 : (_temp_var_721 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_720 == 0 ? indices.field_0 : (_temp_var_720 == 1 ? indices.field_1 : (_temp_var_720 == 2 ? indices.field_2 : (_temp_var_720 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_719 == 0 ? indices.field_0 : (_temp_var_719 == 1 ? indices.field_1 : (_temp_var_719 == 2 ? indices.field_2 : (_temp_var_719 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_483_ is already defined
#ifndef _block_k_483__func
#define _block_k_483__func
__device__ int _block_k_483_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_722 = ((({ int _temp_var_723 = ((({ int _temp_var_724 = ((i % 4));
        (_temp_var_724 == 0 ? indices.field_0 : (_temp_var_724 == 1 ? indices.field_1 : (_temp_var_724 == 2 ? indices.field_2 : (_temp_var_724 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_723 == 0 ? indices.field_0 : (_temp_var_723 == 1 ? indices.field_1 : (_temp_var_723 == 2 ? indices.field_2 : (_temp_var_723 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_722 == 0 ? indices.field_0 : (_temp_var_722 == 1 ? indices.field_1 : (_temp_var_722 == 2 ? indices.field_2 : (_temp_var_722 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_485_ is already defined
#ifndef _block_k_485__func
#define _block_k_485__func
__device__ int _block_k_485_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_725 = ((({ int _temp_var_726 = ((({ int _temp_var_727 = ((i % 4));
        (_temp_var_727 == 0 ? indices.field_0 : (_temp_var_727 == 1 ? indices.field_1 : (_temp_var_727 == 2 ? indices.field_2 : (_temp_var_727 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_726 == 0 ? indices.field_0 : (_temp_var_726 == 1 ? indices.field_1 : (_temp_var_726 == 2 ? indices.field_2 : (_temp_var_726 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_725 == 0 ? indices.field_0 : (_temp_var_725 == 1 ? indices.field_1 : (_temp_var_725 == 2 ? indices.field_2 : (_temp_var_725 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_487_ is already defined
#ifndef _block_k_487__func
#define _block_k_487__func
__device__ int _block_k_487_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_728 = ((({ int _temp_var_729 = ((({ int _temp_var_730 = ((i % 4));
        (_temp_var_730 == 0 ? indices.field_0 : (_temp_var_730 == 1 ? indices.field_1 : (_temp_var_730 == 2 ? indices.field_2 : (_temp_var_730 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_729 == 0 ? indices.field_0 : (_temp_var_729 == 1 ? indices.field_1 : (_temp_var_729 == 2 ? indices.field_2 : (_temp_var_729 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_728 == 0 ? indices.field_0 : (_temp_var_728 == 1 ? indices.field_1 : (_temp_var_728 == 2 ? indices.field_2 : (_temp_var_728 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_489_ is already defined
#ifndef _block_k_489__func
#define _block_k_489__func
__device__ int _block_k_489_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_731 = ((({ int _temp_var_732 = ((({ int _temp_var_733 = ((i % 4));
        (_temp_var_733 == 0 ? indices.field_0 : (_temp_var_733 == 1 ? indices.field_1 : (_temp_var_733 == 2 ? indices.field_2 : (_temp_var_733 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_732 == 0 ? indices.field_0 : (_temp_var_732 == 1 ? indices.field_1 : (_temp_var_732 == 2 ? indices.field_2 : (_temp_var_732 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_731 == 0 ? indices.field_0 : (_temp_var_731 == 1 ? indices.field_1 : (_temp_var_731 == 2 ? indices.field_2 : (_temp_var_731 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_491_ is already defined
#ifndef _block_k_491__func
#define _block_k_491__func
__device__ int _block_k_491_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_734 = ((({ int _temp_var_735 = ((({ int _temp_var_736 = ((i % 4));
        (_temp_var_736 == 0 ? indices.field_0 : (_temp_var_736 == 1 ? indices.field_1 : (_temp_var_736 == 2 ? indices.field_2 : (_temp_var_736 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_735 == 0 ? indices.field_0 : (_temp_var_735 == 1 ? indices.field_1 : (_temp_var_735 == 2 ? indices.field_2 : (_temp_var_735 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_734 == 0 ? indices.field_0 : (_temp_var_734 == 1 ? indices.field_1 : (_temp_var_734 == 2 ? indices.field_2 : (_temp_var_734 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_493_ is already defined
#ifndef _block_k_493__func
#define _block_k_493__func
__device__ int _block_k_493_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_737 = ((({ int _temp_var_738 = ((({ int _temp_var_739 = ((i % 4));
        (_temp_var_739 == 0 ? indices.field_0 : (_temp_var_739 == 1 ? indices.field_1 : (_temp_var_739 == 2 ? indices.field_2 : (_temp_var_739 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_738 == 0 ? indices.field_0 : (_temp_var_738 == 1 ? indices.field_1 : (_temp_var_738 == 2 ? indices.field_2 : (_temp_var_738 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_737 == 0 ? indices.field_0 : (_temp_var_737 == 1 ? indices.field_1 : (_temp_var_737 == 2 ? indices.field_2 : (_temp_var_737 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_495_ is already defined
#ifndef _block_k_495__func
#define _block_k_495__func
__device__ int _block_k_495_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_740 = ((({ int _temp_var_741 = ((({ int _temp_var_742 = ((i % 4));
        (_temp_var_742 == 0 ? indices.field_0 : (_temp_var_742 == 1 ? indices.field_1 : (_temp_var_742 == 2 ? indices.field_2 : (_temp_var_742 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_741 == 0 ? indices.field_0 : (_temp_var_741 == 1 ? indices.field_1 : (_temp_var_741 == 2 ? indices.field_2 : (_temp_var_741 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_740 == 0 ? indices.field_0 : (_temp_var_740 == 1 ? indices.field_1 : (_temp_var_740 == 2 ? indices.field_2 : (_temp_var_740 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_497_ is already defined
#ifndef _block_k_497__func
#define _block_k_497__func
__device__ int _block_k_497_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_743 = ((({ int _temp_var_744 = ((({ int _temp_var_745 = ((i % 4));
        (_temp_var_745 == 0 ? indices.field_0 : (_temp_var_745 == 1 ? indices.field_1 : (_temp_var_745 == 2 ? indices.field_2 : (_temp_var_745 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_744 == 0 ? indices.field_0 : (_temp_var_744 == 1 ? indices.field_1 : (_temp_var_744 == 2 ? indices.field_2 : (_temp_var_744 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_743 == 0 ? indices.field_0 : (_temp_var_743 == 1 ? indices.field_1 : (_temp_var_743 == 2 ? indices.field_2 : (_temp_var_743 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_499_ is already defined
#ifndef _block_k_499__func
#define _block_k_499__func
__device__ int _block_k_499_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_746 = ((({ int _temp_var_747 = ((({ int _temp_var_748 = ((i % 4));
        (_temp_var_748 == 0 ? indices.field_0 : (_temp_var_748 == 1 ? indices.field_1 : (_temp_var_748 == 2 ? indices.field_2 : (_temp_var_748 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_747 == 0 ? indices.field_0 : (_temp_var_747 == 1 ? indices.field_1 : (_temp_var_747 == 2 ? indices.field_2 : (_temp_var_747 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_746 == 0 ? indices.field_0 : (_temp_var_746 == 1 ? indices.field_1 : (_temp_var_746 == 2 ? indices.field_2 : (_temp_var_746 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_501_ is already defined
#ifndef _block_k_501__func
#define _block_k_501__func
__device__ int _block_k_501_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_749 = ((({ int _temp_var_750 = ((({ int _temp_var_751 = ((i % 4));
        (_temp_var_751 == 0 ? indices.field_0 : (_temp_var_751 == 1 ? indices.field_1 : (_temp_var_751 == 2 ? indices.field_2 : (_temp_var_751 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_750 == 0 ? indices.field_0 : (_temp_var_750 == 1 ? indices.field_1 : (_temp_var_750 == 2 ? indices.field_2 : (_temp_var_750 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_749 == 0 ? indices.field_0 : (_temp_var_749 == 1 ? indices.field_1 : (_temp_var_749 == 2 ? indices.field_2 : (_temp_var_749 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_503_ is already defined
#ifndef _block_k_503__func
#define _block_k_503__func
__device__ int _block_k_503_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_752 = ((({ int _temp_var_753 = ((({ int _temp_var_754 = ((i % 4));
        (_temp_var_754 == 0 ? indices.field_0 : (_temp_var_754 == 1 ? indices.field_1 : (_temp_var_754 == 2 ? indices.field_2 : (_temp_var_754 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_753 == 0 ? indices.field_0 : (_temp_var_753 == 1 ? indices.field_1 : (_temp_var_753 == 2 ? indices.field_2 : (_temp_var_753 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_752 == 0 ? indices.field_0 : (_temp_var_752 == 1 ? indices.field_1 : (_temp_var_752 == 2 ? indices.field_2 : (_temp_var_752 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_505_ is already defined
#ifndef _block_k_505__func
#define _block_k_505__func
__device__ int _block_k_505_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_755 = ((({ int _temp_var_756 = ((({ int _temp_var_757 = ((i % 4));
        (_temp_var_757 == 0 ? indices.field_0 : (_temp_var_757 == 1 ? indices.field_1 : (_temp_var_757 == 2 ? indices.field_2 : (_temp_var_757 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_756 == 0 ? indices.field_0 : (_temp_var_756 == 1 ? indices.field_1 : (_temp_var_756 == 2 ? indices.field_2 : (_temp_var_756 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_755 == 0 ? indices.field_0 : (_temp_var_755 == 1 ? indices.field_1 : (_temp_var_755 == 2 ? indices.field_2 : (_temp_var_755 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_507_ is already defined
#ifndef _block_k_507__func
#define _block_k_507__func
__device__ int _block_k_507_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_758 = ((({ int _temp_var_759 = ((({ int _temp_var_760 = ((i % 4));
        (_temp_var_760 == 0 ? indices.field_0 : (_temp_var_760 == 1 ? indices.field_1 : (_temp_var_760 == 2 ? indices.field_2 : (_temp_var_760 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_759 == 0 ? indices.field_0 : (_temp_var_759 == 1 ? indices.field_1 : (_temp_var_759 == 2 ? indices.field_2 : (_temp_var_759 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_758 == 0 ? indices.field_0 : (_temp_var_758 == 1 ? indices.field_1 : (_temp_var_758 == 2 ? indices.field_2 : (_temp_var_758 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_509_ is already defined
#ifndef _block_k_509__func
#define _block_k_509__func
__device__ int _block_k_509_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_761 = ((({ int _temp_var_762 = ((({ int _temp_var_763 = ((i % 4));
        (_temp_var_763 == 0 ? indices.field_0 : (_temp_var_763 == 1 ? indices.field_1 : (_temp_var_763 == 2 ? indices.field_2 : (_temp_var_763 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_762 == 0 ? indices.field_0 : (_temp_var_762 == 1 ? indices.field_1 : (_temp_var_762 == 2 ? indices.field_2 : (_temp_var_762 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_761 == 0 ? indices.field_0 : (_temp_var_761 == 1 ? indices.field_1 : (_temp_var_761 == 2 ? indices.field_2 : (_temp_var_761 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_511_ is already defined
#ifndef _block_k_511__func
#define _block_k_511__func
__device__ int _block_k_511_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_764 = ((({ int _temp_var_765 = ((({ int _temp_var_766 = ((i % 4));
        (_temp_var_766 == 0 ? indices.field_0 : (_temp_var_766 == 1 ? indices.field_1 : (_temp_var_766 == 2 ? indices.field_2 : (_temp_var_766 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_765 == 0 ? indices.field_0 : (_temp_var_765 == 1 ? indices.field_1 : (_temp_var_765 == 2 ? indices.field_2 : (_temp_var_765 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_764 == 0 ? indices.field_0 : (_temp_var_764 == 1 ? indices.field_1 : (_temp_var_764 == 2 ? indices.field_2 : (_temp_var_764 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_513_ is already defined
#ifndef _block_k_513__func
#define _block_k_513__func
__device__ int _block_k_513_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_767 = ((({ int _temp_var_768 = ((({ int _temp_var_769 = ((i % 4));
        (_temp_var_769 == 0 ? indices.field_0 : (_temp_var_769 == 1 ? indices.field_1 : (_temp_var_769 == 2 ? indices.field_2 : (_temp_var_769 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_768 == 0 ? indices.field_0 : (_temp_var_768 == 1 ? indices.field_1 : (_temp_var_768 == 2 ? indices.field_2 : (_temp_var_768 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_767 == 0 ? indices.field_0 : (_temp_var_767 == 1 ? indices.field_1 : (_temp_var_767 == 2 ? indices.field_2 : (_temp_var_767 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_515_ is already defined
#ifndef _block_k_515__func
#define _block_k_515__func
__device__ int _block_k_515_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_770 = ((({ int _temp_var_771 = ((({ int _temp_var_772 = ((i % 4));
        (_temp_var_772 == 0 ? indices.field_0 : (_temp_var_772 == 1 ? indices.field_1 : (_temp_var_772 == 2 ? indices.field_2 : (_temp_var_772 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_771 == 0 ? indices.field_0 : (_temp_var_771 == 1 ? indices.field_1 : (_temp_var_771 == 2 ? indices.field_2 : (_temp_var_771 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_770 == 0 ? indices.field_0 : (_temp_var_770 == 1 ? indices.field_1 : (_temp_var_770 == 2 ? indices.field_2 : (_temp_var_770 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_517_ is already defined
#ifndef _block_k_517__func
#define _block_k_517__func
__device__ int _block_k_517_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_773 = ((({ int _temp_var_774 = ((({ int _temp_var_775 = ((i % 4));
        (_temp_var_775 == 0 ? indices.field_0 : (_temp_var_775 == 1 ? indices.field_1 : (_temp_var_775 == 2 ? indices.field_2 : (_temp_var_775 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_774 == 0 ? indices.field_0 : (_temp_var_774 == 1 ? indices.field_1 : (_temp_var_774 == 2 ? indices.field_2 : (_temp_var_774 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_773 == 0 ? indices.field_0 : (_temp_var_773 == 1 ? indices.field_1 : (_temp_var_773 == 2 ? indices.field_2 : (_temp_var_773 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_519_ is already defined
#ifndef _block_k_519__func
#define _block_k_519__func
__device__ int _block_k_519_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_776 = ((({ int _temp_var_777 = ((({ int _temp_var_778 = ((i % 4));
        (_temp_var_778 == 0 ? indices.field_0 : (_temp_var_778 == 1 ? indices.field_1 : (_temp_var_778 == 2 ? indices.field_2 : (_temp_var_778 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_777 == 0 ? indices.field_0 : (_temp_var_777 == 1 ? indices.field_1 : (_temp_var_777 == 2 ? indices.field_2 : (_temp_var_777 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_776 == 0 ? indices.field_0 : (_temp_var_776 == 1 ? indices.field_1 : (_temp_var_776 == 2 ? indices.field_2 : (_temp_var_776 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_521_ is already defined
#ifndef _block_k_521__func
#define _block_k_521__func
__device__ int _block_k_521_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_779 = ((({ int _temp_var_780 = ((({ int _temp_var_781 = ((i % 4));
        (_temp_var_781 == 0 ? indices.field_0 : (_temp_var_781 == 1 ? indices.field_1 : (_temp_var_781 == 2 ? indices.field_2 : (_temp_var_781 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_780 == 0 ? indices.field_0 : (_temp_var_780 == 1 ? indices.field_1 : (_temp_var_780 == 2 ? indices.field_2 : (_temp_var_780 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_779 == 0 ? indices.field_0 : (_temp_var_779 == 1 ? indices.field_1 : (_temp_var_779 == 2 ? indices.field_2 : (_temp_var_779 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_523_ is already defined
#ifndef _block_k_523__func
#define _block_k_523__func
__device__ int _block_k_523_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_782 = ((({ int _temp_var_783 = ((({ int _temp_var_784 = ((i % 4));
        (_temp_var_784 == 0 ? indices.field_0 : (_temp_var_784 == 1 ? indices.field_1 : (_temp_var_784 == 2 ? indices.field_2 : (_temp_var_784 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_783 == 0 ? indices.field_0 : (_temp_var_783 == 1 ? indices.field_1 : (_temp_var_783 == 2 ? indices.field_2 : (_temp_var_783 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_782 == 0 ? indices.field_0 : (_temp_var_782 == 1 ? indices.field_1 : (_temp_var_782 == 2 ? indices.field_2 : (_temp_var_782 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_525_ is already defined
#ifndef _block_k_525__func
#define _block_k_525__func
__device__ int _block_k_525_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_785 = ((({ int _temp_var_786 = ((({ int _temp_var_787 = ((i % 4));
        (_temp_var_787 == 0 ? indices.field_0 : (_temp_var_787 == 1 ? indices.field_1 : (_temp_var_787 == 2 ? indices.field_2 : (_temp_var_787 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_786 == 0 ? indices.field_0 : (_temp_var_786 == 1 ? indices.field_1 : (_temp_var_786 == 2 ? indices.field_2 : (_temp_var_786 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_785 == 0 ? indices.field_0 : (_temp_var_785 == 1 ? indices.field_1 : (_temp_var_785 == 2 ? indices.field_2 : (_temp_var_785 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_527_ is already defined
#ifndef _block_k_527__func
#define _block_k_527__func
__device__ int _block_k_527_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_788 = ((({ int _temp_var_789 = ((({ int _temp_var_790 = ((i % 4));
        (_temp_var_790 == 0 ? indices.field_0 : (_temp_var_790 == 1 ? indices.field_1 : (_temp_var_790 == 2 ? indices.field_2 : (_temp_var_790 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_789 == 0 ? indices.field_0 : (_temp_var_789 == 1 ? indices.field_1 : (_temp_var_789 == 2 ? indices.field_2 : (_temp_var_789 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_788 == 0 ? indices.field_0 : (_temp_var_788 == 1 ? indices.field_1 : (_temp_var_788 == 2 ? indices.field_2 : (_temp_var_788 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_529_ is already defined
#ifndef _block_k_529__func
#define _block_k_529__func
__device__ int _block_k_529_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_791 = ((({ int _temp_var_792 = ((({ int _temp_var_793 = ((i % 4));
        (_temp_var_793 == 0 ? indices.field_0 : (_temp_var_793 == 1 ? indices.field_1 : (_temp_var_793 == 2 ? indices.field_2 : (_temp_var_793 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_792 == 0 ? indices.field_0 : (_temp_var_792 == 1 ? indices.field_1 : (_temp_var_792 == 2 ? indices.field_2 : (_temp_var_792 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_791 == 0 ? indices.field_0 : (_temp_var_791 == 1 ? indices.field_1 : (_temp_var_791 == 2 ? indices.field_2 : (_temp_var_791 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_531_ is already defined
#ifndef _block_k_531__func
#define _block_k_531__func
__device__ int _block_k_531_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_794 = ((({ int _temp_var_795 = ((({ int _temp_var_796 = ((i % 4));
        (_temp_var_796 == 0 ? indices.field_0 : (_temp_var_796 == 1 ? indices.field_1 : (_temp_var_796 == 2 ? indices.field_2 : (_temp_var_796 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_795 == 0 ? indices.field_0 : (_temp_var_795 == 1 ? indices.field_1 : (_temp_var_795 == 2 ? indices.field_2 : (_temp_var_795 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_794 == 0 ? indices.field_0 : (_temp_var_794 == 1 ? indices.field_1 : (_temp_var_794 == 2 ? indices.field_2 : (_temp_var_794 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_533_ is already defined
#ifndef _block_k_533__func
#define _block_k_533__func
__device__ int _block_k_533_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_797 = ((({ int _temp_var_798 = ((({ int _temp_var_799 = ((i % 4));
        (_temp_var_799 == 0 ? indices.field_0 : (_temp_var_799 == 1 ? indices.field_1 : (_temp_var_799 == 2 ? indices.field_2 : (_temp_var_799 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_798 == 0 ? indices.field_0 : (_temp_var_798 == 1 ? indices.field_1 : (_temp_var_798 == 2 ? indices.field_2 : (_temp_var_798 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_797 == 0 ? indices.field_0 : (_temp_var_797 == 1 ? indices.field_1 : (_temp_var_797 == 2 ? indices.field_2 : (_temp_var_797 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_535_ is already defined
#ifndef _block_k_535__func
#define _block_k_535__func
__device__ int _block_k_535_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_800 = ((({ int _temp_var_801 = ((({ int _temp_var_802 = ((i % 4));
        (_temp_var_802 == 0 ? indices.field_0 : (_temp_var_802 == 1 ? indices.field_1 : (_temp_var_802 == 2 ? indices.field_2 : (_temp_var_802 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_801 == 0 ? indices.field_0 : (_temp_var_801 == 1 ? indices.field_1 : (_temp_var_801 == 2 ? indices.field_2 : (_temp_var_801 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_800 == 0 ? indices.field_0 : (_temp_var_800 == 1 ? indices.field_1 : (_temp_var_800 == 2 ? indices.field_2 : (_temp_var_800 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_537_ is already defined
#ifndef _block_k_537__func
#define _block_k_537__func
__device__ int _block_k_537_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_803 = ((({ int _temp_var_804 = ((({ int _temp_var_805 = ((i % 4));
        (_temp_var_805 == 0 ? indices.field_0 : (_temp_var_805 == 1 ? indices.field_1 : (_temp_var_805 == 2 ? indices.field_2 : (_temp_var_805 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_804 == 0 ? indices.field_0 : (_temp_var_804 == 1 ? indices.field_1 : (_temp_var_804 == 2 ? indices.field_2 : (_temp_var_804 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_803 == 0 ? indices.field_0 : (_temp_var_803 == 1 ? indices.field_1 : (_temp_var_803 == 2 ? indices.field_2 : (_temp_var_803 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_539_ is already defined
#ifndef _block_k_539__func
#define _block_k_539__func
__device__ int _block_k_539_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_806 = ((({ int _temp_var_807 = ((({ int _temp_var_808 = ((i % 4));
        (_temp_var_808 == 0 ? indices.field_0 : (_temp_var_808 == 1 ? indices.field_1 : (_temp_var_808 == 2 ? indices.field_2 : (_temp_var_808 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_807 == 0 ? indices.field_0 : (_temp_var_807 == 1 ? indices.field_1 : (_temp_var_807 == 2 ? indices.field_2 : (_temp_var_807 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_806 == 0 ? indices.field_0 : (_temp_var_806 == 1 ? indices.field_1 : (_temp_var_806 == 2 ? indices.field_2 : (_temp_var_806 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_541_ is already defined
#ifndef _block_k_541__func
#define _block_k_541__func
__device__ int _block_k_541_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_809 = ((({ int _temp_var_810 = ((({ int _temp_var_811 = ((i % 4));
        (_temp_var_811 == 0 ? indices.field_0 : (_temp_var_811 == 1 ? indices.field_1 : (_temp_var_811 == 2 ? indices.field_2 : (_temp_var_811 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_810 == 0 ? indices.field_0 : (_temp_var_810 == 1 ? indices.field_1 : (_temp_var_810 == 2 ? indices.field_2 : (_temp_var_810 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_809 == 0 ? indices.field_0 : (_temp_var_809 == 1 ? indices.field_1 : (_temp_var_809 == 2 ? indices.field_2 : (_temp_var_809 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_543_ is already defined
#ifndef _block_k_543__func
#define _block_k_543__func
__device__ int _block_k_543_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_812 = ((({ int _temp_var_813 = ((({ int _temp_var_814 = ((i % 4));
        (_temp_var_814 == 0 ? indices.field_0 : (_temp_var_814 == 1 ? indices.field_1 : (_temp_var_814 == 2 ? indices.field_2 : (_temp_var_814 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_813 == 0 ? indices.field_0 : (_temp_var_813 == 1 ? indices.field_1 : (_temp_var_813 == 2 ? indices.field_2 : (_temp_var_813 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_812 == 0 ? indices.field_0 : (_temp_var_812 == 1 ? indices.field_1 : (_temp_var_812 == 2 ? indices.field_2 : (_temp_var_812 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_545_ is already defined
#ifndef _block_k_545__func
#define _block_k_545__func
__device__ int _block_k_545_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_815 = ((({ int _temp_var_816 = ((({ int _temp_var_817 = ((i % 4));
        (_temp_var_817 == 0 ? indices.field_0 : (_temp_var_817 == 1 ? indices.field_1 : (_temp_var_817 == 2 ? indices.field_2 : (_temp_var_817 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_816 == 0 ? indices.field_0 : (_temp_var_816 == 1 ? indices.field_1 : (_temp_var_816 == 2 ? indices.field_2 : (_temp_var_816 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_815 == 0 ? indices.field_0 : (_temp_var_815 == 1 ? indices.field_1 : (_temp_var_815 == 2 ? indices.field_2 : (_temp_var_815 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_547_ is already defined
#ifndef _block_k_547__func
#define _block_k_547__func
__device__ int _block_k_547_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_818 = ((({ int _temp_var_819 = ((({ int _temp_var_820 = ((i % 4));
        (_temp_var_820 == 0 ? indices.field_0 : (_temp_var_820 == 1 ? indices.field_1 : (_temp_var_820 == 2 ? indices.field_2 : (_temp_var_820 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_819 == 0 ? indices.field_0 : (_temp_var_819 == 1 ? indices.field_1 : (_temp_var_819 == 2 ? indices.field_2 : (_temp_var_819 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_818 == 0 ? indices.field_0 : (_temp_var_818 == 1 ? indices.field_1 : (_temp_var_818 == 2 ? indices.field_2 : (_temp_var_818 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_549_ is already defined
#ifndef _block_k_549__func
#define _block_k_549__func
__device__ int _block_k_549_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_821 = ((({ int _temp_var_822 = ((({ int _temp_var_823 = ((i % 4));
        (_temp_var_823 == 0 ? indices.field_0 : (_temp_var_823 == 1 ? indices.field_1 : (_temp_var_823 == 2 ? indices.field_2 : (_temp_var_823 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_822 == 0 ? indices.field_0 : (_temp_var_822 == 1 ? indices.field_1 : (_temp_var_822 == 2 ? indices.field_2 : (_temp_var_822 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_821 == 0 ? indices.field_0 : (_temp_var_821 == 1 ? indices.field_1 : (_temp_var_821 == 2 ? indices.field_2 : (_temp_var_821 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_551_ is already defined
#ifndef _block_k_551__func
#define _block_k_551__func
__device__ int _block_k_551_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_824 = ((({ int _temp_var_825 = ((({ int _temp_var_826 = ((i % 4));
        (_temp_var_826 == 0 ? indices.field_0 : (_temp_var_826 == 1 ? indices.field_1 : (_temp_var_826 == 2 ? indices.field_2 : (_temp_var_826 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_825 == 0 ? indices.field_0 : (_temp_var_825 == 1 ? indices.field_1 : (_temp_var_825 == 2 ? indices.field_2 : (_temp_var_825 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_824 == 0 ? indices.field_0 : (_temp_var_824 == 1 ? indices.field_1 : (_temp_var_824 == 2 ? indices.field_2 : (_temp_var_824 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_553_ is already defined
#ifndef _block_k_553__func
#define _block_k_553__func
__device__ int _block_k_553_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_827 = ((({ int _temp_var_828 = ((({ int _temp_var_829 = ((i % 4));
        (_temp_var_829 == 0 ? indices.field_0 : (_temp_var_829 == 1 ? indices.field_1 : (_temp_var_829 == 2 ? indices.field_2 : (_temp_var_829 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_828 == 0 ? indices.field_0 : (_temp_var_828 == 1 ? indices.field_1 : (_temp_var_828 == 2 ? indices.field_2 : (_temp_var_828 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_827 == 0 ? indices.field_0 : (_temp_var_827 == 1 ? indices.field_1 : (_temp_var_827 == 2 ? indices.field_2 : (_temp_var_827 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_555_ is already defined
#ifndef _block_k_555__func
#define _block_k_555__func
__device__ int _block_k_555_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_830 = ((({ int _temp_var_831 = ((({ int _temp_var_832 = ((i % 4));
        (_temp_var_832 == 0 ? indices.field_0 : (_temp_var_832 == 1 ? indices.field_1 : (_temp_var_832 == 2 ? indices.field_2 : (_temp_var_832 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_831 == 0 ? indices.field_0 : (_temp_var_831 == 1 ? indices.field_1 : (_temp_var_831 == 2 ? indices.field_2 : (_temp_var_831 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_830 == 0 ? indices.field_0 : (_temp_var_830 == 1 ? indices.field_1 : (_temp_var_830 == 2 ? indices.field_2 : (_temp_var_830 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_557_ is already defined
#ifndef _block_k_557__func
#define _block_k_557__func
__device__ int _block_k_557_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_833 = ((({ int _temp_var_834 = ((({ int _temp_var_835 = ((i % 4));
        (_temp_var_835 == 0 ? indices.field_0 : (_temp_var_835 == 1 ? indices.field_1 : (_temp_var_835 == 2 ? indices.field_2 : (_temp_var_835 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_834 == 0 ? indices.field_0 : (_temp_var_834 == 1 ? indices.field_1 : (_temp_var_834 == 2 ? indices.field_2 : (_temp_var_834 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_833 == 0 ? indices.field_0 : (_temp_var_833 == 1 ? indices.field_1 : (_temp_var_833 == 2 ? indices.field_2 : (_temp_var_833 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_559_ is already defined
#ifndef _block_k_559__func
#define _block_k_559__func
__device__ int _block_k_559_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_836 = ((({ int _temp_var_837 = ((({ int _temp_var_838 = ((i % 4));
        (_temp_var_838 == 0 ? indices.field_0 : (_temp_var_838 == 1 ? indices.field_1 : (_temp_var_838 == 2 ? indices.field_2 : (_temp_var_838 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_837 == 0 ? indices.field_0 : (_temp_var_837 == 1 ? indices.field_1 : (_temp_var_837 == 2 ? indices.field_2 : (_temp_var_837 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_836 == 0 ? indices.field_0 : (_temp_var_836 == 1 ? indices.field_1 : (_temp_var_836 == 2 ? indices.field_2 : (_temp_var_836 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_561_ is already defined
#ifndef _block_k_561__func
#define _block_k_561__func
__device__ int _block_k_561_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_839 = ((({ int _temp_var_840 = ((({ int _temp_var_841 = ((i % 4));
        (_temp_var_841 == 0 ? indices.field_0 : (_temp_var_841 == 1 ? indices.field_1 : (_temp_var_841 == 2 ? indices.field_2 : (_temp_var_841 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_840 == 0 ? indices.field_0 : (_temp_var_840 == 1 ? indices.field_1 : (_temp_var_840 == 2 ? indices.field_2 : (_temp_var_840 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_839 == 0 ? indices.field_0 : (_temp_var_839 == 1 ? indices.field_1 : (_temp_var_839 == 2 ? indices.field_2 : (_temp_var_839 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_563_ is already defined
#ifndef _block_k_563__func
#define _block_k_563__func
__device__ int _block_k_563_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_842 = ((({ int _temp_var_843 = ((({ int _temp_var_844 = ((i % 4));
        (_temp_var_844 == 0 ? indices.field_0 : (_temp_var_844 == 1 ? indices.field_1 : (_temp_var_844 == 2 ? indices.field_2 : (_temp_var_844 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_843 == 0 ? indices.field_0 : (_temp_var_843 == 1 ? indices.field_1 : (_temp_var_843 == 2 ? indices.field_2 : (_temp_var_843 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_842 == 0 ? indices.field_0 : (_temp_var_842 == 1 ? indices.field_1 : (_temp_var_842 == 2 ? indices.field_2 : (_temp_var_842 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_565_ is already defined
#ifndef _block_k_565__func
#define _block_k_565__func
__device__ int _block_k_565_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_845 = ((({ int _temp_var_846 = ((({ int _temp_var_847 = ((i % 4));
        (_temp_var_847 == 0 ? indices.field_0 : (_temp_var_847 == 1 ? indices.field_1 : (_temp_var_847 == 2 ? indices.field_2 : (_temp_var_847 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_846 == 0 ? indices.field_0 : (_temp_var_846 == 1 ? indices.field_1 : (_temp_var_846 == 2 ? indices.field_2 : (_temp_var_846 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_845 == 0 ? indices.field_0 : (_temp_var_845 == 1 ? indices.field_1 : (_temp_var_845 == 2 ? indices.field_2 : (_temp_var_845 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_567_ is already defined
#ifndef _block_k_567__func
#define _block_k_567__func
__device__ int _block_k_567_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_848 = ((({ int _temp_var_849 = ((({ int _temp_var_850 = ((i % 4));
        (_temp_var_850 == 0 ? indices.field_0 : (_temp_var_850 == 1 ? indices.field_1 : (_temp_var_850 == 2 ? indices.field_2 : (_temp_var_850 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_849 == 0 ? indices.field_0 : (_temp_var_849 == 1 ? indices.field_1 : (_temp_var_849 == 2 ? indices.field_2 : (_temp_var_849 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_848 == 0 ? indices.field_0 : (_temp_var_848 == 1 ? indices.field_1 : (_temp_var_848 == 2 ? indices.field_2 : (_temp_var_848 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_569_ is already defined
#ifndef _block_k_569__func
#define _block_k_569__func
__device__ int _block_k_569_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_851 = ((({ int _temp_var_852 = ((({ int _temp_var_853 = ((i % 4));
        (_temp_var_853 == 0 ? indices.field_0 : (_temp_var_853 == 1 ? indices.field_1 : (_temp_var_853 == 2 ? indices.field_2 : (_temp_var_853 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_852 == 0 ? indices.field_0 : (_temp_var_852 == 1 ? indices.field_1 : (_temp_var_852 == 2 ? indices.field_2 : (_temp_var_852 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_851 == 0 ? indices.field_0 : (_temp_var_851 == 1 ? indices.field_1 : (_temp_var_851 == 2 ? indices.field_2 : (_temp_var_851 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_571_ is already defined
#ifndef _block_k_571__func
#define _block_k_571__func
__device__ int _block_k_571_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_854 = ((({ int _temp_var_855 = ((({ int _temp_var_856 = ((i % 4));
        (_temp_var_856 == 0 ? indices.field_0 : (_temp_var_856 == 1 ? indices.field_1 : (_temp_var_856 == 2 ? indices.field_2 : (_temp_var_856 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_855 == 0 ? indices.field_0 : (_temp_var_855 == 1 ? indices.field_1 : (_temp_var_855 == 2 ? indices.field_2 : (_temp_var_855 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_854 == 0 ? indices.field_0 : (_temp_var_854 == 1 ? indices.field_1 : (_temp_var_854 == 2 ? indices.field_2 : (_temp_var_854 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_573_ is already defined
#ifndef _block_k_573__func
#define _block_k_573__func
__device__ int _block_k_573_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_857 = ((({ int _temp_var_858 = ((({ int _temp_var_859 = ((i % 4));
        (_temp_var_859 == 0 ? indices.field_0 : (_temp_var_859 == 1 ? indices.field_1 : (_temp_var_859 == 2 ? indices.field_2 : (_temp_var_859 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_858 == 0 ? indices.field_0 : (_temp_var_858 == 1 ? indices.field_1 : (_temp_var_858 == 2 ? indices.field_2 : (_temp_var_858 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_857 == 0 ? indices.field_0 : (_temp_var_857 == 1 ? indices.field_1 : (_temp_var_857 == 2 ? indices.field_2 : (_temp_var_857 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_575_ is already defined
#ifndef _block_k_575__func
#define _block_k_575__func
__device__ int _block_k_575_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_860 = ((({ int _temp_var_861 = ((({ int _temp_var_862 = ((i % 4));
        (_temp_var_862 == 0 ? indices.field_0 : (_temp_var_862 == 1 ? indices.field_1 : (_temp_var_862 == 2 ? indices.field_2 : (_temp_var_862 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_861 == 0 ? indices.field_0 : (_temp_var_861 == 1 ? indices.field_1 : (_temp_var_861 == 2 ? indices.field_2 : (_temp_var_861 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_860 == 0 ? indices.field_0 : (_temp_var_860 == 1 ? indices.field_1 : (_temp_var_860 == 2 ? indices.field_2 : (_temp_var_860 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_577_ is already defined
#ifndef _block_k_577__func
#define _block_k_577__func
__device__ int _block_k_577_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_863 = ((({ int _temp_var_864 = ((({ int _temp_var_865 = ((i % 4));
        (_temp_var_865 == 0 ? indices.field_0 : (_temp_var_865 == 1 ? indices.field_1 : (_temp_var_865 == 2 ? indices.field_2 : (_temp_var_865 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_864 == 0 ? indices.field_0 : (_temp_var_864 == 1 ? indices.field_1 : (_temp_var_864 == 2 ? indices.field_2 : (_temp_var_864 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_863 == 0 ? indices.field_0 : (_temp_var_863 == 1 ? indices.field_1 : (_temp_var_863 == 2 ? indices.field_2 : (_temp_var_863 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_579_ is already defined
#ifndef _block_k_579__func
#define _block_k_579__func
__device__ int _block_k_579_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_866 = ((({ int _temp_var_867 = ((({ int _temp_var_868 = ((i % 4));
        (_temp_var_868 == 0 ? indices.field_0 : (_temp_var_868 == 1 ? indices.field_1 : (_temp_var_868 == 2 ? indices.field_2 : (_temp_var_868 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_867 == 0 ? indices.field_0 : (_temp_var_867 == 1 ? indices.field_1 : (_temp_var_867 == 2 ? indices.field_2 : (_temp_var_867 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_866 == 0 ? indices.field_0 : (_temp_var_866 == 1 ? indices.field_1 : (_temp_var_866 == 2 ? indices.field_2 : (_temp_var_866 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_581_ is already defined
#ifndef _block_k_581__func
#define _block_k_581__func
__device__ int _block_k_581_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_869 = ((({ int _temp_var_870 = ((({ int _temp_var_871 = ((i % 4));
        (_temp_var_871 == 0 ? indices.field_0 : (_temp_var_871 == 1 ? indices.field_1 : (_temp_var_871 == 2 ? indices.field_2 : (_temp_var_871 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_870 == 0 ? indices.field_0 : (_temp_var_870 == 1 ? indices.field_1 : (_temp_var_870 == 2 ? indices.field_2 : (_temp_var_870 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_869 == 0 ? indices.field_0 : (_temp_var_869 == 1 ? indices.field_1 : (_temp_var_869 == 2 ? indices.field_2 : (_temp_var_869 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_583_ is already defined
#ifndef _block_k_583__func
#define _block_k_583__func
__device__ int _block_k_583_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_872 = ((({ int _temp_var_873 = ((({ int _temp_var_874 = ((i % 4));
        (_temp_var_874 == 0 ? indices.field_0 : (_temp_var_874 == 1 ? indices.field_1 : (_temp_var_874 == 2 ? indices.field_2 : (_temp_var_874 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_873 == 0 ? indices.field_0 : (_temp_var_873 == 1 ? indices.field_1 : (_temp_var_873 == 2 ? indices.field_2 : (_temp_var_873 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_872 == 0 ? indices.field_0 : (_temp_var_872 == 1 ? indices.field_1 : (_temp_var_872 == 2 ? indices.field_2 : (_temp_var_872 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_585_ is already defined
#ifndef _block_k_585__func
#define _block_k_585__func
__device__ int _block_k_585_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_875 = ((({ int _temp_var_876 = ((({ int _temp_var_877 = ((i % 4));
        (_temp_var_877 == 0 ? indices.field_0 : (_temp_var_877 == 1 ? indices.field_1 : (_temp_var_877 == 2 ? indices.field_2 : (_temp_var_877 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_876 == 0 ? indices.field_0 : (_temp_var_876 == 1 ? indices.field_1 : (_temp_var_876 == 2 ? indices.field_2 : (_temp_var_876 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_875 == 0 ? indices.field_0 : (_temp_var_875 == 1 ? indices.field_1 : (_temp_var_875 == 2 ? indices.field_2 : (_temp_var_875 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_587_ is already defined
#ifndef _block_k_587__func
#define _block_k_587__func
__device__ int _block_k_587_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_878 = ((({ int _temp_var_879 = ((({ int _temp_var_880 = ((i % 4));
        (_temp_var_880 == 0 ? indices.field_0 : (_temp_var_880 == 1 ? indices.field_1 : (_temp_var_880 == 2 ? indices.field_2 : (_temp_var_880 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_879 == 0 ? indices.field_0 : (_temp_var_879 == 1 ? indices.field_1 : (_temp_var_879 == 2 ? indices.field_2 : (_temp_var_879 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_878 == 0 ? indices.field_0 : (_temp_var_878 == 1 ? indices.field_1 : (_temp_var_878 == 2 ? indices.field_2 : (_temp_var_878 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_589_ is already defined
#ifndef _block_k_589__func
#define _block_k_589__func
__device__ int _block_k_589_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_881 = ((({ int _temp_var_882 = ((({ int _temp_var_883 = ((i % 4));
        (_temp_var_883 == 0 ? indices.field_0 : (_temp_var_883 == 1 ? indices.field_1 : (_temp_var_883 == 2 ? indices.field_2 : (_temp_var_883 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_882 == 0 ? indices.field_0 : (_temp_var_882 == 1 ? indices.field_1 : (_temp_var_882 == 2 ? indices.field_2 : (_temp_var_882 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_881 == 0 ? indices.field_0 : (_temp_var_881 == 1 ? indices.field_1 : (_temp_var_881 == 2 ? indices.field_2 : (_temp_var_881 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_591_ is already defined
#ifndef _block_k_591__func
#define _block_k_591__func
__device__ int _block_k_591_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_884 = ((({ int _temp_var_885 = ((({ int _temp_var_886 = ((i % 4));
        (_temp_var_886 == 0 ? indices.field_0 : (_temp_var_886 == 1 ? indices.field_1 : (_temp_var_886 == 2 ? indices.field_2 : (_temp_var_886 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_885 == 0 ? indices.field_0 : (_temp_var_885 == 1 ? indices.field_1 : (_temp_var_885 == 2 ? indices.field_2 : (_temp_var_885 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_884 == 0 ? indices.field_0 : (_temp_var_884 == 1 ? indices.field_1 : (_temp_var_884 == 2 ? indices.field_2 : (_temp_var_884 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_593_ is already defined
#ifndef _block_k_593__func
#define _block_k_593__func
__device__ int _block_k_593_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_887 = ((({ int _temp_var_888 = ((({ int _temp_var_889 = ((i % 4));
        (_temp_var_889 == 0 ? indices.field_0 : (_temp_var_889 == 1 ? indices.field_1 : (_temp_var_889 == 2 ? indices.field_2 : (_temp_var_889 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_888 == 0 ? indices.field_0 : (_temp_var_888 == 1 ? indices.field_1 : (_temp_var_888 == 2 ? indices.field_2 : (_temp_var_888 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_887 == 0 ? indices.field_0 : (_temp_var_887 == 1 ? indices.field_1 : (_temp_var_887 == 2 ? indices.field_2 : (_temp_var_887 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_595_ is already defined
#ifndef _block_k_595__func
#define _block_k_595__func
__device__ int _block_k_595_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_890 = ((({ int _temp_var_891 = ((({ int _temp_var_892 = ((i % 4));
        (_temp_var_892 == 0 ? indices.field_0 : (_temp_var_892 == 1 ? indices.field_1 : (_temp_var_892 == 2 ? indices.field_2 : (_temp_var_892 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_891 == 0 ? indices.field_0 : (_temp_var_891 == 1 ? indices.field_1 : (_temp_var_891 == 2 ? indices.field_2 : (_temp_var_891 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_890 == 0 ? indices.field_0 : (_temp_var_890 == 1 ? indices.field_1 : (_temp_var_890 == 2 ? indices.field_2 : (_temp_var_890 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_597_ is already defined
#ifndef _block_k_597__func
#define _block_k_597__func
__device__ int _block_k_597_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_893 = ((({ int _temp_var_894 = ((({ int _temp_var_895 = ((i % 4));
        (_temp_var_895 == 0 ? indices.field_0 : (_temp_var_895 == 1 ? indices.field_1 : (_temp_var_895 == 2 ? indices.field_2 : (_temp_var_895 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_894 == 0 ? indices.field_0 : (_temp_var_894 == 1 ? indices.field_1 : (_temp_var_894 == 2 ? indices.field_2 : (_temp_var_894 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_893 == 0 ? indices.field_0 : (_temp_var_893 == 1 ? indices.field_1 : (_temp_var_893 == 2 ? indices.field_2 : (_temp_var_893 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_599_ is already defined
#ifndef _block_k_599__func
#define _block_k_599__func
__device__ int _block_k_599_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_896 = ((({ int _temp_var_897 = ((({ int _temp_var_898 = ((i % 4));
        (_temp_var_898 == 0 ? indices.field_0 : (_temp_var_898 == 1 ? indices.field_1 : (_temp_var_898 == 2 ? indices.field_2 : (_temp_var_898 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_897 == 0 ? indices.field_0 : (_temp_var_897 == 1 ? indices.field_1 : (_temp_var_897 == 2 ? indices.field_2 : (_temp_var_897 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_896 == 0 ? indices.field_0 : (_temp_var_896 == 1 ? indices.field_1 : (_temp_var_896 == 2 ? indices.field_2 : (_temp_var_896 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_601_ is already defined
#ifndef _block_k_601__func
#define _block_k_601__func
__device__ int _block_k_601_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_899 = ((({ int _temp_var_900 = ((({ int _temp_var_901 = ((i % 4));
        (_temp_var_901 == 0 ? indices.field_0 : (_temp_var_901 == 1 ? indices.field_1 : (_temp_var_901 == 2 ? indices.field_2 : (_temp_var_901 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_900 == 0 ? indices.field_0 : (_temp_var_900 == 1 ? indices.field_1 : (_temp_var_900 == 2 ? indices.field_2 : (_temp_var_900 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_899 == 0 ? indices.field_0 : (_temp_var_899 == 1 ? indices.field_1 : (_temp_var_899 == 2 ? indices.field_2 : (_temp_var_899 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_603_ is already defined
#ifndef _block_k_603__func
#define _block_k_603__func
__device__ int _block_k_603_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_902 = ((({ int _temp_var_903 = ((({ int _temp_var_904 = ((i % 4));
        (_temp_var_904 == 0 ? indices.field_0 : (_temp_var_904 == 1 ? indices.field_1 : (_temp_var_904 == 2 ? indices.field_2 : (_temp_var_904 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_903 == 0 ? indices.field_0 : (_temp_var_903 == 1 ? indices.field_1 : (_temp_var_903 == 2 ? indices.field_2 : (_temp_var_903 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_902 == 0 ? indices.field_0 : (_temp_var_902 == 1 ? indices.field_1 : (_temp_var_902 == 2 ? indices.field_2 : (_temp_var_902 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_605_ is already defined
#ifndef _block_k_605__func
#define _block_k_605__func
__device__ int _block_k_605_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_905 = ((({ int _temp_var_906 = ((({ int _temp_var_907 = ((i % 4));
        (_temp_var_907 == 0 ? indices.field_0 : (_temp_var_907 == 1 ? indices.field_1 : (_temp_var_907 == 2 ? indices.field_2 : (_temp_var_907 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_906 == 0 ? indices.field_0 : (_temp_var_906 == 1 ? indices.field_1 : (_temp_var_906 == 2 ? indices.field_2 : (_temp_var_906 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_905 == 0 ? indices.field_0 : (_temp_var_905 == 1 ? indices.field_1 : (_temp_var_905 == 2 ? indices.field_2 : (_temp_var_905 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_607_ is already defined
#ifndef _block_k_607__func
#define _block_k_607__func
__device__ int _block_k_607_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_908 = ((({ int _temp_var_909 = ((({ int _temp_var_910 = ((i % 4));
        (_temp_var_910 == 0 ? indices.field_0 : (_temp_var_910 == 1 ? indices.field_1 : (_temp_var_910 == 2 ? indices.field_2 : (_temp_var_910 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_909 == 0 ? indices.field_0 : (_temp_var_909 == 1 ? indices.field_1 : (_temp_var_909 == 2 ? indices.field_2 : (_temp_var_909 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_908 == 0 ? indices.field_0 : (_temp_var_908 == 1 ? indices.field_1 : (_temp_var_908 == 2 ? indices.field_2 : (_temp_var_908 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_609_ is already defined
#ifndef _block_k_609__func
#define _block_k_609__func
__device__ int _block_k_609_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_911 = ((({ int _temp_var_912 = ((({ int _temp_var_913 = ((i % 4));
        (_temp_var_913 == 0 ? indices.field_0 : (_temp_var_913 == 1 ? indices.field_1 : (_temp_var_913 == 2 ? indices.field_2 : (_temp_var_913 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_912 == 0 ? indices.field_0 : (_temp_var_912 == 1 ? indices.field_1 : (_temp_var_912 == 2 ? indices.field_2 : (_temp_var_912 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_911 == 0 ? indices.field_0 : (_temp_var_911 == 1 ? indices.field_1 : (_temp_var_911 == 2 ? indices.field_2 : (_temp_var_911 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_611_ is already defined
#ifndef _block_k_611__func
#define _block_k_611__func
__device__ int _block_k_611_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_914 = ((({ int _temp_var_915 = ((({ int _temp_var_916 = ((i % 4));
        (_temp_var_916 == 0 ? indices.field_0 : (_temp_var_916 == 1 ? indices.field_1 : (_temp_var_916 == 2 ? indices.field_2 : (_temp_var_916 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_915 == 0 ? indices.field_0 : (_temp_var_915 == 1 ? indices.field_1 : (_temp_var_915 == 2 ? indices.field_2 : (_temp_var_915 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_914 == 0 ? indices.field_0 : (_temp_var_914 == 1 ? indices.field_1 : (_temp_var_914 == 2 ? indices.field_2 : (_temp_var_914 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_613_ is already defined
#ifndef _block_k_613__func
#define _block_k_613__func
__device__ int _block_k_613_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_917 = ((({ int _temp_var_918 = ((({ int _temp_var_919 = ((i % 4));
        (_temp_var_919 == 0 ? indices.field_0 : (_temp_var_919 == 1 ? indices.field_1 : (_temp_var_919 == 2 ? indices.field_2 : (_temp_var_919 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_918 == 0 ? indices.field_0 : (_temp_var_918 == 1 ? indices.field_1 : (_temp_var_918 == 2 ? indices.field_2 : (_temp_var_918 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_917 == 0 ? indices.field_0 : (_temp_var_917 == 1 ? indices.field_1 : (_temp_var_917 == 2 ? indices.field_2 : (_temp_var_917 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_615_ is already defined
#ifndef _block_k_615__func
#define _block_k_615__func
__device__ int _block_k_615_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_920 = ((({ int _temp_var_921 = ((({ int _temp_var_922 = ((i % 4));
        (_temp_var_922 == 0 ? indices.field_0 : (_temp_var_922 == 1 ? indices.field_1 : (_temp_var_922 == 2 ? indices.field_2 : (_temp_var_922 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_921 == 0 ? indices.field_0 : (_temp_var_921 == 1 ? indices.field_1 : (_temp_var_921 == 2 ? indices.field_2 : (_temp_var_921 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_920 == 0 ? indices.field_0 : (_temp_var_920 == 1 ? indices.field_1 : (_temp_var_920 == 2 ? indices.field_2 : (_temp_var_920 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_617_ is already defined
#ifndef _block_k_617__func
#define _block_k_617__func
__device__ int _block_k_617_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_923 = ((({ int _temp_var_924 = ((({ int _temp_var_925 = ((i % 4));
        (_temp_var_925 == 0 ? indices.field_0 : (_temp_var_925 == 1 ? indices.field_1 : (_temp_var_925 == 2 ? indices.field_2 : (_temp_var_925 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_924 == 0 ? indices.field_0 : (_temp_var_924 == 1 ? indices.field_1 : (_temp_var_924 == 2 ? indices.field_2 : (_temp_var_924 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_923 == 0 ? indices.field_0 : (_temp_var_923 == 1 ? indices.field_1 : (_temp_var_923 == 2 ? indices.field_2 : (_temp_var_923 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_619_ is already defined
#ifndef _block_k_619__func
#define _block_k_619__func
__device__ int _block_k_619_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_926 = ((({ int _temp_var_927 = ((({ int _temp_var_928 = ((i % 4));
        (_temp_var_928 == 0 ? indices.field_0 : (_temp_var_928 == 1 ? indices.field_1 : (_temp_var_928 == 2 ? indices.field_2 : (_temp_var_928 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_927 == 0 ? indices.field_0 : (_temp_var_927 == 1 ? indices.field_1 : (_temp_var_927 == 2 ? indices.field_2 : (_temp_var_927 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_926 == 0 ? indices.field_0 : (_temp_var_926 == 1 ? indices.field_1 : (_temp_var_926 == 2 ? indices.field_2 : (_temp_var_926 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_621_ is already defined
#ifndef _block_k_621__func
#define _block_k_621__func
__device__ int _block_k_621_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_929 = ((({ int _temp_var_930 = ((({ int _temp_var_931 = ((i % 4));
        (_temp_var_931 == 0 ? indices.field_0 : (_temp_var_931 == 1 ? indices.field_1 : (_temp_var_931 == 2 ? indices.field_2 : (_temp_var_931 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_930 == 0 ? indices.field_0 : (_temp_var_930 == 1 ? indices.field_1 : (_temp_var_930 == 2 ? indices.field_2 : (_temp_var_930 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_929 == 0 ? indices.field_0 : (_temp_var_929 == 1 ? indices.field_1 : (_temp_var_929 == 2 ? indices.field_2 : (_temp_var_929 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_623_ is already defined
#ifndef _block_k_623__func
#define _block_k_623__func
__device__ int _block_k_623_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_932 = ((({ int _temp_var_933 = ((({ int _temp_var_934 = ((i % 4));
        (_temp_var_934 == 0 ? indices.field_0 : (_temp_var_934 == 1 ? indices.field_1 : (_temp_var_934 == 2 ? indices.field_2 : (_temp_var_934 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_933 == 0 ? indices.field_0 : (_temp_var_933 == 1 ? indices.field_1 : (_temp_var_933 == 2 ? indices.field_2 : (_temp_var_933 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_932 == 0 ? indices.field_0 : (_temp_var_932 == 1 ? indices.field_1 : (_temp_var_932 == 2 ? indices.field_2 : (_temp_var_932 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_625_ is already defined
#ifndef _block_k_625__func
#define _block_k_625__func
__device__ int _block_k_625_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_935 = ((({ int _temp_var_936 = ((({ int _temp_var_937 = ((i % 4));
        (_temp_var_937 == 0 ? indices.field_0 : (_temp_var_937 == 1 ? indices.field_1 : (_temp_var_937 == 2 ? indices.field_2 : (_temp_var_937 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_936 == 0 ? indices.field_0 : (_temp_var_936 == 1 ? indices.field_1 : (_temp_var_936 == 2 ? indices.field_2 : (_temp_var_936 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_935 == 0 ? indices.field_0 : (_temp_var_935 == 1 ? indices.field_1 : (_temp_var_935 == 2 ? indices.field_2 : (_temp_var_935 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_627_ is already defined
#ifndef _block_k_627__func
#define _block_k_627__func
__device__ int _block_k_627_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_938 = ((({ int _temp_var_939 = ((({ int _temp_var_940 = ((i % 4));
        (_temp_var_940 == 0 ? indices.field_0 : (_temp_var_940 == 1 ? indices.field_1 : (_temp_var_940 == 2 ? indices.field_2 : (_temp_var_940 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_939 == 0 ? indices.field_0 : (_temp_var_939 == 1 ? indices.field_1 : (_temp_var_939 == 2 ? indices.field_2 : (_temp_var_939 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_938 == 0 ? indices.field_0 : (_temp_var_938 == 1 ? indices.field_1 : (_temp_var_938 == 2 ? indices.field_2 : (_temp_var_938 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_629_ is already defined
#ifndef _block_k_629__func
#define _block_k_629__func
__device__ int _block_k_629_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_941 = ((({ int _temp_var_942 = ((({ int _temp_var_943 = ((i % 4));
        (_temp_var_943 == 0 ? indices.field_0 : (_temp_var_943 == 1 ? indices.field_1 : (_temp_var_943 == 2 ? indices.field_2 : (_temp_var_943 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_942 == 0 ? indices.field_0 : (_temp_var_942 == 1 ? indices.field_1 : (_temp_var_942 == 2 ? indices.field_2 : (_temp_var_942 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_941 == 0 ? indices.field_0 : (_temp_var_941 == 1 ? indices.field_1 : (_temp_var_941 == 2 ? indices.field_2 : (_temp_var_941 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_631_ is already defined
#ifndef _block_k_631__func
#define _block_k_631__func
__device__ int _block_k_631_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_944 = ((({ int _temp_var_945 = ((({ int _temp_var_946 = ((i % 4));
        (_temp_var_946 == 0 ? indices.field_0 : (_temp_var_946 == 1 ? indices.field_1 : (_temp_var_946 == 2 ? indices.field_2 : (_temp_var_946 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_945 == 0 ? indices.field_0 : (_temp_var_945 == 1 ? indices.field_1 : (_temp_var_945 == 2 ? indices.field_2 : (_temp_var_945 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_944 == 0 ? indices.field_0 : (_temp_var_944 == 1 ? indices.field_1 : (_temp_var_944 == 2 ? indices.field_2 : (_temp_var_944 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_633_ is already defined
#ifndef _block_k_633__func
#define _block_k_633__func
__device__ int _block_k_633_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_947 = ((({ int _temp_var_948 = ((({ int _temp_var_949 = ((i % 4));
        (_temp_var_949 == 0 ? indices.field_0 : (_temp_var_949 == 1 ? indices.field_1 : (_temp_var_949 == 2 ? indices.field_2 : (_temp_var_949 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_948 == 0 ? indices.field_0 : (_temp_var_948 == 1 ? indices.field_1 : (_temp_var_948 == 2 ? indices.field_2 : (_temp_var_948 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_947 == 0 ? indices.field_0 : (_temp_var_947 == 1 ? indices.field_1 : (_temp_var_947 == 2 ? indices.field_2 : (_temp_var_947 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_635_ is already defined
#ifndef _block_k_635__func
#define _block_k_635__func
__device__ int _block_k_635_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_950 = ((({ int _temp_var_951 = ((({ int _temp_var_952 = ((i % 4));
        (_temp_var_952 == 0 ? indices.field_0 : (_temp_var_952 == 1 ? indices.field_1 : (_temp_var_952 == 2 ? indices.field_2 : (_temp_var_952 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_951 == 0 ? indices.field_0 : (_temp_var_951 == 1 ? indices.field_1 : (_temp_var_951 == 2 ? indices.field_2 : (_temp_var_951 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_950 == 0 ? indices.field_0 : (_temp_var_950 == 1 ? indices.field_1 : (_temp_var_950 == 2 ? indices.field_2 : (_temp_var_950 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_637_ is already defined
#ifndef _block_k_637__func
#define _block_k_637__func
__device__ int _block_k_637_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_953 = ((({ int _temp_var_954 = ((({ int _temp_var_955 = ((i % 4));
        (_temp_var_955 == 0 ? indices.field_0 : (_temp_var_955 == 1 ? indices.field_1 : (_temp_var_955 == 2 ? indices.field_2 : (_temp_var_955 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_954 == 0 ? indices.field_0 : (_temp_var_954 == 1 ? indices.field_1 : (_temp_var_954 == 2 ? indices.field_2 : (_temp_var_954 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_953 == 0 ? indices.field_0 : (_temp_var_953 == 1 ? indices.field_1 : (_temp_var_953 == 2 ? indices.field_2 : (_temp_var_953 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_639_ is already defined
#ifndef _block_k_639__func
#define _block_k_639__func
__device__ int _block_k_639_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_956 = ((({ int _temp_var_957 = ((({ int _temp_var_958 = ((i % 4));
        (_temp_var_958 == 0 ? indices.field_0 : (_temp_var_958 == 1 ? indices.field_1 : (_temp_var_958 == 2 ? indices.field_2 : (_temp_var_958 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_957 == 0 ? indices.field_0 : (_temp_var_957 == 1 ? indices.field_1 : (_temp_var_957 == 2 ? indices.field_2 : (_temp_var_957 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_956 == 0 ? indices.field_0 : (_temp_var_956 == 1 ? indices.field_1 : (_temp_var_956 == 2 ? indices.field_2 : (_temp_var_956 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_641_ is already defined
#ifndef _block_k_641__func
#define _block_k_641__func
__device__ int _block_k_641_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_959 = ((({ int _temp_var_960 = ((({ int _temp_var_961 = ((i % 4));
        (_temp_var_961 == 0 ? indices.field_0 : (_temp_var_961 == 1 ? indices.field_1 : (_temp_var_961 == 2 ? indices.field_2 : (_temp_var_961 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_960 == 0 ? indices.field_0 : (_temp_var_960 == 1 ? indices.field_1 : (_temp_var_960 == 2 ? indices.field_2 : (_temp_var_960 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_959 == 0 ? indices.field_0 : (_temp_var_959 == 1 ? indices.field_1 : (_temp_var_959 == 2 ? indices.field_2 : (_temp_var_959 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_643_ is already defined
#ifndef _block_k_643__func
#define _block_k_643__func
__device__ int _block_k_643_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_962 = ((({ int _temp_var_963 = ((({ int _temp_var_964 = ((i % 4));
        (_temp_var_964 == 0 ? indices.field_0 : (_temp_var_964 == 1 ? indices.field_1 : (_temp_var_964 == 2 ? indices.field_2 : (_temp_var_964 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_963 == 0 ? indices.field_0 : (_temp_var_963 == 1 ? indices.field_1 : (_temp_var_963 == 2 ? indices.field_2 : (_temp_var_963 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_962 == 0 ? indices.field_0 : (_temp_var_962 == 1 ? indices.field_1 : (_temp_var_962 == 2 ? indices.field_2 : (_temp_var_962 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_645_ is already defined
#ifndef _block_k_645__func
#define _block_k_645__func
__device__ int _block_k_645_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_965 = ((({ int _temp_var_966 = ((({ int _temp_var_967 = ((i % 4));
        (_temp_var_967 == 0 ? indices.field_0 : (_temp_var_967 == 1 ? indices.field_1 : (_temp_var_967 == 2 ? indices.field_2 : (_temp_var_967 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_966 == 0 ? indices.field_0 : (_temp_var_966 == 1 ? indices.field_1 : (_temp_var_966 == 2 ? indices.field_2 : (_temp_var_966 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_965 == 0 ? indices.field_0 : (_temp_var_965 == 1 ? indices.field_1 : (_temp_var_965 == 2 ? indices.field_2 : (_temp_var_965 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_647_ is already defined
#ifndef _block_k_647__func
#define _block_k_647__func
__device__ int _block_k_647_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_968 = ((({ int _temp_var_969 = ((({ int _temp_var_970 = ((i % 4));
        (_temp_var_970 == 0 ? indices.field_0 : (_temp_var_970 == 1 ? indices.field_1 : (_temp_var_970 == 2 ? indices.field_2 : (_temp_var_970 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_969 == 0 ? indices.field_0 : (_temp_var_969 == 1 ? indices.field_1 : (_temp_var_969 == 2 ? indices.field_2 : (_temp_var_969 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_968 == 0 ? indices.field_0 : (_temp_var_968 == 1 ? indices.field_1 : (_temp_var_968 == 2 ? indices.field_2 : (_temp_var_968 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_649_ is already defined
#ifndef _block_k_649__func
#define _block_k_649__func
__device__ int _block_k_649_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_971 = ((({ int _temp_var_972 = ((({ int _temp_var_973 = ((i % 4));
        (_temp_var_973 == 0 ? indices.field_0 : (_temp_var_973 == 1 ? indices.field_1 : (_temp_var_973 == 2 ? indices.field_2 : (_temp_var_973 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_972 == 0 ? indices.field_0 : (_temp_var_972 == 1 ? indices.field_1 : (_temp_var_972 == 2 ? indices.field_2 : (_temp_var_972 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_971 == 0 ? indices.field_0 : (_temp_var_971 == 1 ? indices.field_1 : (_temp_var_971 == 2 ? indices.field_2 : (_temp_var_971 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_651_ is already defined
#ifndef _block_k_651__func
#define _block_k_651__func
__device__ int _block_k_651_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_974 = ((({ int _temp_var_975 = ((({ int _temp_var_976 = ((i % 4));
        (_temp_var_976 == 0 ? indices.field_0 : (_temp_var_976 == 1 ? indices.field_1 : (_temp_var_976 == 2 ? indices.field_2 : (_temp_var_976 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_975 == 0 ? indices.field_0 : (_temp_var_975 == 1 ? indices.field_1 : (_temp_var_975 == 2 ? indices.field_2 : (_temp_var_975 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_974 == 0 ? indices.field_0 : (_temp_var_974 == 1 ? indices.field_1 : (_temp_var_974 == 2 ? indices.field_2 : (_temp_var_974 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_653_ is already defined
#ifndef _block_k_653__func
#define _block_k_653__func
__device__ int _block_k_653_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_977 = ((({ int _temp_var_978 = ((({ int _temp_var_979 = ((i % 4));
        (_temp_var_979 == 0 ? indices.field_0 : (_temp_var_979 == 1 ? indices.field_1 : (_temp_var_979 == 2 ? indices.field_2 : (_temp_var_979 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_978 == 0 ? indices.field_0 : (_temp_var_978 == 1 ? indices.field_1 : (_temp_var_978 == 2 ? indices.field_2 : (_temp_var_978 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_977 == 0 ? indices.field_0 : (_temp_var_977 == 1 ? indices.field_1 : (_temp_var_977 == 2 ? indices.field_2 : (_temp_var_977 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_655_ is already defined
#ifndef _block_k_655__func
#define _block_k_655__func
__device__ int _block_k_655_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_980 = ((({ int _temp_var_981 = ((({ int _temp_var_982 = ((i % 4));
        (_temp_var_982 == 0 ? indices.field_0 : (_temp_var_982 == 1 ? indices.field_1 : (_temp_var_982 == 2 ? indices.field_2 : (_temp_var_982 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_981 == 0 ? indices.field_0 : (_temp_var_981 == 1 ? indices.field_1 : (_temp_var_981 == 2 ? indices.field_2 : (_temp_var_981 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_980 == 0 ? indices.field_0 : (_temp_var_980 == 1 ? indices.field_1 : (_temp_var_980 == 2 ? indices.field_2 : (_temp_var_980 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_657_ is already defined
#ifndef _block_k_657__func
#define _block_k_657__func
__device__ int _block_k_657_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_983 = ((({ int _temp_var_984 = ((({ int _temp_var_985 = ((i % 4));
        (_temp_var_985 == 0 ? indices.field_0 : (_temp_var_985 == 1 ? indices.field_1 : (_temp_var_985 == 2 ? indices.field_2 : (_temp_var_985 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_984 == 0 ? indices.field_0 : (_temp_var_984 == 1 ? indices.field_1 : (_temp_var_984 == 2 ? indices.field_2 : (_temp_var_984 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_983 == 0 ? indices.field_0 : (_temp_var_983 == 1 ? indices.field_1 : (_temp_var_983 == 2 ? indices.field_2 : (_temp_var_983 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_659_ is already defined
#ifndef _block_k_659__func
#define _block_k_659__func
__device__ int _block_k_659_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_986 = ((({ int _temp_var_987 = ((({ int _temp_var_988 = ((i % 4));
        (_temp_var_988 == 0 ? indices.field_0 : (_temp_var_988 == 1 ? indices.field_1 : (_temp_var_988 == 2 ? indices.field_2 : (_temp_var_988 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_987 == 0 ? indices.field_0 : (_temp_var_987 == 1 ? indices.field_1 : (_temp_var_987 == 2 ? indices.field_2 : (_temp_var_987 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_986 == 0 ? indices.field_0 : (_temp_var_986 == 1 ? indices.field_1 : (_temp_var_986 == 2 ? indices.field_2 : (_temp_var_986 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_661_ is already defined
#ifndef _block_k_661__func
#define _block_k_661__func
__device__ int _block_k_661_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_989 = ((({ int _temp_var_990 = ((({ int _temp_var_991 = ((i % 4));
        (_temp_var_991 == 0 ? indices.field_0 : (_temp_var_991 == 1 ? indices.field_1 : (_temp_var_991 == 2 ? indices.field_2 : (_temp_var_991 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_990 == 0 ? indices.field_0 : (_temp_var_990 == 1 ? indices.field_1 : (_temp_var_990 == 2 ? indices.field_2 : (_temp_var_990 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_989 == 0 ? indices.field_0 : (_temp_var_989 == 1 ? indices.field_1 : (_temp_var_989 == 2 ? indices.field_2 : (_temp_var_989 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_663_ is already defined
#ifndef _block_k_663__func
#define _block_k_663__func
__device__ int _block_k_663_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_992 = ((({ int _temp_var_993 = ((({ int _temp_var_994 = ((i % 4));
        (_temp_var_994 == 0 ? indices.field_0 : (_temp_var_994 == 1 ? indices.field_1 : (_temp_var_994 == 2 ? indices.field_2 : (_temp_var_994 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_993 == 0 ? indices.field_0 : (_temp_var_993 == 1 ? indices.field_1 : (_temp_var_993 == 2 ? indices.field_2 : (_temp_var_993 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_992 == 0 ? indices.field_0 : (_temp_var_992 == 1 ? indices.field_1 : (_temp_var_992 == 2 ? indices.field_2 : (_temp_var_992 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_665_ is already defined
#ifndef _block_k_665__func
#define _block_k_665__func
__device__ int _block_k_665_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_995 = ((({ int _temp_var_996 = ((({ int _temp_var_997 = ((i % 4));
        (_temp_var_997 == 0 ? indices.field_0 : (_temp_var_997 == 1 ? indices.field_1 : (_temp_var_997 == 2 ? indices.field_2 : (_temp_var_997 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_996 == 0 ? indices.field_0 : (_temp_var_996 == 1 ? indices.field_1 : (_temp_var_996 == 2 ? indices.field_2 : (_temp_var_996 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_995 == 0 ? indices.field_0 : (_temp_var_995 == 1 ? indices.field_1 : (_temp_var_995 == 2 ? indices.field_2 : (_temp_var_995 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_667_ is already defined
#ifndef _block_k_667__func
#define _block_k_667__func
__device__ int _block_k_667_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_998 = ((({ int _temp_var_999 = ((({ int _temp_var_1000 = ((i % 4));
        (_temp_var_1000 == 0 ? indices.field_0 : (_temp_var_1000 == 1 ? indices.field_1 : (_temp_var_1000 == 2 ? indices.field_2 : (_temp_var_1000 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_999 == 0 ? indices.field_0 : (_temp_var_999 == 1 ? indices.field_1 : (_temp_var_999 == 2 ? indices.field_2 : (_temp_var_999 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_998 == 0 ? indices.field_0 : (_temp_var_998 == 1 ? indices.field_1 : (_temp_var_998 == 2 ? indices.field_2 : (_temp_var_998 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_669_ is already defined
#ifndef _block_k_669__func
#define _block_k_669__func
__device__ int _block_k_669_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1001 = ((({ int _temp_var_1002 = ((({ int _temp_var_1003 = ((i % 4));
        (_temp_var_1003 == 0 ? indices.field_0 : (_temp_var_1003 == 1 ? indices.field_1 : (_temp_var_1003 == 2 ? indices.field_2 : (_temp_var_1003 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1002 == 0 ? indices.field_0 : (_temp_var_1002 == 1 ? indices.field_1 : (_temp_var_1002 == 2 ? indices.field_2 : (_temp_var_1002 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1001 == 0 ? indices.field_0 : (_temp_var_1001 == 1 ? indices.field_1 : (_temp_var_1001 == 2 ? indices.field_2 : (_temp_var_1001 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_671_ is already defined
#ifndef _block_k_671__func
#define _block_k_671__func
__device__ int _block_k_671_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1004 = ((({ int _temp_var_1005 = ((({ int _temp_var_1006 = ((i % 4));
        (_temp_var_1006 == 0 ? indices.field_0 : (_temp_var_1006 == 1 ? indices.field_1 : (_temp_var_1006 == 2 ? indices.field_2 : (_temp_var_1006 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1005 == 0 ? indices.field_0 : (_temp_var_1005 == 1 ? indices.field_1 : (_temp_var_1005 == 2 ? indices.field_2 : (_temp_var_1005 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1004 == 0 ? indices.field_0 : (_temp_var_1004 == 1 ? indices.field_1 : (_temp_var_1004 == 2 ? indices.field_2 : (_temp_var_1004 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_673_ is already defined
#ifndef _block_k_673__func
#define _block_k_673__func
__device__ int _block_k_673_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1007 = ((({ int _temp_var_1008 = ((({ int _temp_var_1009 = ((i % 4));
        (_temp_var_1009 == 0 ? indices.field_0 : (_temp_var_1009 == 1 ? indices.field_1 : (_temp_var_1009 == 2 ? indices.field_2 : (_temp_var_1009 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1008 == 0 ? indices.field_0 : (_temp_var_1008 == 1 ? indices.field_1 : (_temp_var_1008 == 2 ? indices.field_2 : (_temp_var_1008 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1007 == 0 ? indices.field_0 : (_temp_var_1007 == 1 ? indices.field_1 : (_temp_var_1007 == 2 ? indices.field_2 : (_temp_var_1007 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_675_ is already defined
#ifndef _block_k_675__func
#define _block_k_675__func
__device__ int _block_k_675_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1010 = ((({ int _temp_var_1011 = ((({ int _temp_var_1012 = ((i % 4));
        (_temp_var_1012 == 0 ? indices.field_0 : (_temp_var_1012 == 1 ? indices.field_1 : (_temp_var_1012 == 2 ? indices.field_2 : (_temp_var_1012 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1011 == 0 ? indices.field_0 : (_temp_var_1011 == 1 ? indices.field_1 : (_temp_var_1011 == 2 ? indices.field_2 : (_temp_var_1011 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1010 == 0 ? indices.field_0 : (_temp_var_1010 == 1 ? indices.field_1 : (_temp_var_1010 == 2 ? indices.field_2 : (_temp_var_1010 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_677_ is already defined
#ifndef _block_k_677__func
#define _block_k_677__func
__device__ int _block_k_677_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1013 = ((({ int _temp_var_1014 = ((({ int _temp_var_1015 = ((i % 4));
        (_temp_var_1015 == 0 ? indices.field_0 : (_temp_var_1015 == 1 ? indices.field_1 : (_temp_var_1015 == 2 ? indices.field_2 : (_temp_var_1015 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1014 == 0 ? indices.field_0 : (_temp_var_1014 == 1 ? indices.field_1 : (_temp_var_1014 == 2 ? indices.field_2 : (_temp_var_1014 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1013 == 0 ? indices.field_0 : (_temp_var_1013 == 1 ? indices.field_1 : (_temp_var_1013 == 2 ? indices.field_2 : (_temp_var_1013 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_679_ is already defined
#ifndef _block_k_679__func
#define _block_k_679__func
__device__ int _block_k_679_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1016 = ((({ int _temp_var_1017 = ((({ int _temp_var_1018 = ((i % 4));
        (_temp_var_1018 == 0 ? indices.field_0 : (_temp_var_1018 == 1 ? indices.field_1 : (_temp_var_1018 == 2 ? indices.field_2 : (_temp_var_1018 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1017 == 0 ? indices.field_0 : (_temp_var_1017 == 1 ? indices.field_1 : (_temp_var_1017 == 2 ? indices.field_2 : (_temp_var_1017 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1016 == 0 ? indices.field_0 : (_temp_var_1016 == 1 ? indices.field_1 : (_temp_var_1016 == 2 ? indices.field_2 : (_temp_var_1016 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_681_ is already defined
#ifndef _block_k_681__func
#define _block_k_681__func
__device__ int _block_k_681_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1019 = ((({ int _temp_var_1020 = ((({ int _temp_var_1021 = ((i % 4));
        (_temp_var_1021 == 0 ? indices.field_0 : (_temp_var_1021 == 1 ? indices.field_1 : (_temp_var_1021 == 2 ? indices.field_2 : (_temp_var_1021 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1020 == 0 ? indices.field_0 : (_temp_var_1020 == 1 ? indices.field_1 : (_temp_var_1020 == 2 ? indices.field_2 : (_temp_var_1020 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1019 == 0 ? indices.field_0 : (_temp_var_1019 == 1 ? indices.field_1 : (_temp_var_1019 == 2 ? indices.field_2 : (_temp_var_1019 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_683_ is already defined
#ifndef _block_k_683__func
#define _block_k_683__func
__device__ int _block_k_683_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1022 = ((({ int _temp_var_1023 = ((({ int _temp_var_1024 = ((i % 4));
        (_temp_var_1024 == 0 ? indices.field_0 : (_temp_var_1024 == 1 ? indices.field_1 : (_temp_var_1024 == 2 ? indices.field_2 : (_temp_var_1024 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1023 == 0 ? indices.field_0 : (_temp_var_1023 == 1 ? indices.field_1 : (_temp_var_1023 == 2 ? indices.field_2 : (_temp_var_1023 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1022 == 0 ? indices.field_0 : (_temp_var_1022 == 1 ? indices.field_1 : (_temp_var_1022 == 2 ? indices.field_2 : (_temp_var_1022 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_685_ is already defined
#ifndef _block_k_685__func
#define _block_k_685__func
__device__ int _block_k_685_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1025 = ((({ int _temp_var_1026 = ((({ int _temp_var_1027 = ((i % 4));
        (_temp_var_1027 == 0 ? indices.field_0 : (_temp_var_1027 == 1 ? indices.field_1 : (_temp_var_1027 == 2 ? indices.field_2 : (_temp_var_1027 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1026 == 0 ? indices.field_0 : (_temp_var_1026 == 1 ? indices.field_1 : (_temp_var_1026 == 2 ? indices.field_2 : (_temp_var_1026 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1025 == 0 ? indices.field_0 : (_temp_var_1025 == 1 ? indices.field_1 : (_temp_var_1025 == 2 ? indices.field_2 : (_temp_var_1025 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_687_ is already defined
#ifndef _block_k_687__func
#define _block_k_687__func
__device__ int _block_k_687_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1028 = ((({ int _temp_var_1029 = ((({ int _temp_var_1030 = ((i % 4));
        (_temp_var_1030 == 0 ? indices.field_0 : (_temp_var_1030 == 1 ? indices.field_1 : (_temp_var_1030 == 2 ? indices.field_2 : (_temp_var_1030 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1029 == 0 ? indices.field_0 : (_temp_var_1029 == 1 ? indices.field_1 : (_temp_var_1029 == 2 ? indices.field_2 : (_temp_var_1029 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1028 == 0 ? indices.field_0 : (_temp_var_1028 == 1 ? indices.field_1 : (_temp_var_1028 == 2 ? indices.field_2 : (_temp_var_1028 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_689_ is already defined
#ifndef _block_k_689__func
#define _block_k_689__func
__device__ int _block_k_689_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1031 = ((({ int _temp_var_1032 = ((({ int _temp_var_1033 = ((i % 4));
        (_temp_var_1033 == 0 ? indices.field_0 : (_temp_var_1033 == 1 ? indices.field_1 : (_temp_var_1033 == 2 ? indices.field_2 : (_temp_var_1033 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1032 == 0 ? indices.field_0 : (_temp_var_1032 == 1 ? indices.field_1 : (_temp_var_1032 == 2 ? indices.field_2 : (_temp_var_1032 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1031 == 0 ? indices.field_0 : (_temp_var_1031 == 1 ? indices.field_1 : (_temp_var_1031 == 2 ? indices.field_2 : (_temp_var_1031 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_691_ is already defined
#ifndef _block_k_691__func
#define _block_k_691__func
__device__ int _block_k_691_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1034 = ((({ int _temp_var_1035 = ((({ int _temp_var_1036 = ((i % 4));
        (_temp_var_1036 == 0 ? indices.field_0 : (_temp_var_1036 == 1 ? indices.field_1 : (_temp_var_1036 == 2 ? indices.field_2 : (_temp_var_1036 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1035 == 0 ? indices.field_0 : (_temp_var_1035 == 1 ? indices.field_1 : (_temp_var_1035 == 2 ? indices.field_2 : (_temp_var_1035 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1034 == 0 ? indices.field_0 : (_temp_var_1034 == 1 ? indices.field_1 : (_temp_var_1034 == 2 ? indices.field_2 : (_temp_var_1034 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_693_ is already defined
#ifndef _block_k_693__func
#define _block_k_693__func
__device__ int _block_k_693_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1037 = ((({ int _temp_var_1038 = ((({ int _temp_var_1039 = ((i % 4));
        (_temp_var_1039 == 0 ? indices.field_0 : (_temp_var_1039 == 1 ? indices.field_1 : (_temp_var_1039 == 2 ? indices.field_2 : (_temp_var_1039 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1038 == 0 ? indices.field_0 : (_temp_var_1038 == 1 ? indices.field_1 : (_temp_var_1038 == 2 ? indices.field_2 : (_temp_var_1038 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1037 == 0 ? indices.field_0 : (_temp_var_1037 == 1 ? indices.field_1 : (_temp_var_1037 == 2 ? indices.field_2 : (_temp_var_1037 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_695_ is already defined
#ifndef _block_k_695__func
#define _block_k_695__func
__device__ int _block_k_695_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1040 = ((({ int _temp_var_1041 = ((({ int _temp_var_1042 = ((i % 4));
        (_temp_var_1042 == 0 ? indices.field_0 : (_temp_var_1042 == 1 ? indices.field_1 : (_temp_var_1042 == 2 ? indices.field_2 : (_temp_var_1042 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1041 == 0 ? indices.field_0 : (_temp_var_1041 == 1 ? indices.field_1 : (_temp_var_1041 == 2 ? indices.field_2 : (_temp_var_1041 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1040 == 0 ? indices.field_0 : (_temp_var_1040 == 1 ? indices.field_1 : (_temp_var_1040 == 2 ? indices.field_2 : (_temp_var_1040 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_697_ is already defined
#ifndef _block_k_697__func
#define _block_k_697__func
__device__ int _block_k_697_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1043 = ((({ int _temp_var_1044 = ((({ int _temp_var_1045 = ((i % 4));
        (_temp_var_1045 == 0 ? indices.field_0 : (_temp_var_1045 == 1 ? indices.field_1 : (_temp_var_1045 == 2 ? indices.field_2 : (_temp_var_1045 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1044 == 0 ? indices.field_0 : (_temp_var_1044 == 1 ? indices.field_1 : (_temp_var_1044 == 2 ? indices.field_2 : (_temp_var_1044 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1043 == 0 ? indices.field_0 : (_temp_var_1043 == 1 ? indices.field_1 : (_temp_var_1043 == 2 ? indices.field_2 : (_temp_var_1043 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_699_ is already defined
#ifndef _block_k_699__func
#define _block_k_699__func
__device__ int _block_k_699_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1046 = ((({ int _temp_var_1047 = ((({ int _temp_var_1048 = ((i % 4));
        (_temp_var_1048 == 0 ? indices.field_0 : (_temp_var_1048 == 1 ? indices.field_1 : (_temp_var_1048 == 2 ? indices.field_2 : (_temp_var_1048 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1047 == 0 ? indices.field_0 : (_temp_var_1047 == 1 ? indices.field_1 : (_temp_var_1047 == 2 ? indices.field_2 : (_temp_var_1047 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1046 == 0 ? indices.field_0 : (_temp_var_1046 == 1 ? indices.field_1 : (_temp_var_1046 == 2 ? indices.field_2 : (_temp_var_1046 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_701_ is already defined
#ifndef _block_k_701__func
#define _block_k_701__func
__device__ int _block_k_701_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1049 = ((({ int _temp_var_1050 = ((({ int _temp_var_1051 = ((i % 4));
        (_temp_var_1051 == 0 ? indices.field_0 : (_temp_var_1051 == 1 ? indices.field_1 : (_temp_var_1051 == 2 ? indices.field_2 : (_temp_var_1051 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1050 == 0 ? indices.field_0 : (_temp_var_1050 == 1 ? indices.field_1 : (_temp_var_1050 == 2 ? indices.field_2 : (_temp_var_1050 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1049 == 0 ? indices.field_0 : (_temp_var_1049 == 1 ? indices.field_1 : (_temp_var_1049 == 2 ? indices.field_2 : (_temp_var_1049 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_703_ is already defined
#ifndef _block_k_703__func
#define _block_k_703__func
__device__ int _block_k_703_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1052 = ((({ int _temp_var_1053 = ((({ int _temp_var_1054 = ((i % 4));
        (_temp_var_1054 == 0 ? indices.field_0 : (_temp_var_1054 == 1 ? indices.field_1 : (_temp_var_1054 == 2 ? indices.field_2 : (_temp_var_1054 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1053 == 0 ? indices.field_0 : (_temp_var_1053 == 1 ? indices.field_1 : (_temp_var_1053 == 2 ? indices.field_2 : (_temp_var_1053 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1052 == 0 ? indices.field_0 : (_temp_var_1052 == 1 ? indices.field_1 : (_temp_var_1052 == 2 ? indices.field_2 : (_temp_var_1052 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_705_ is already defined
#ifndef _block_k_705__func
#define _block_k_705__func
__device__ int _block_k_705_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1055 = ((({ int _temp_var_1056 = ((({ int _temp_var_1057 = ((i % 4));
        (_temp_var_1057 == 0 ? indices.field_0 : (_temp_var_1057 == 1 ? indices.field_1 : (_temp_var_1057 == 2 ? indices.field_2 : (_temp_var_1057 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1056 == 0 ? indices.field_0 : (_temp_var_1056 == 1 ? indices.field_1 : (_temp_var_1056 == 2 ? indices.field_2 : (_temp_var_1056 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1055 == 0 ? indices.field_0 : (_temp_var_1055 == 1 ? indices.field_1 : (_temp_var_1055 == 2 ? indices.field_2 : (_temp_var_1055 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_707_ is already defined
#ifndef _block_k_707__func
#define _block_k_707__func
__device__ int _block_k_707_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1058 = ((({ int _temp_var_1059 = ((({ int _temp_var_1060 = ((i % 4));
        (_temp_var_1060 == 0 ? indices.field_0 : (_temp_var_1060 == 1 ? indices.field_1 : (_temp_var_1060 == 2 ? indices.field_2 : (_temp_var_1060 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1059 == 0 ? indices.field_0 : (_temp_var_1059 == 1 ? indices.field_1 : (_temp_var_1059 == 2 ? indices.field_2 : (_temp_var_1059 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1058 == 0 ? indices.field_0 : (_temp_var_1058 == 1 ? indices.field_1 : (_temp_var_1058 == 2 ? indices.field_2 : (_temp_var_1058 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_709_ is already defined
#ifndef _block_k_709__func
#define _block_k_709__func
__device__ int _block_k_709_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1061 = ((({ int _temp_var_1062 = ((({ int _temp_var_1063 = ((i % 4));
        (_temp_var_1063 == 0 ? indices.field_0 : (_temp_var_1063 == 1 ? indices.field_1 : (_temp_var_1063 == 2 ? indices.field_2 : (_temp_var_1063 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1062 == 0 ? indices.field_0 : (_temp_var_1062 == 1 ? indices.field_1 : (_temp_var_1062 == 2 ? indices.field_2 : (_temp_var_1062 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1061 == 0 ? indices.field_0 : (_temp_var_1061 == 1 ? indices.field_1 : (_temp_var_1061 == 2 ? indices.field_2 : (_temp_var_1061 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_711_ is already defined
#ifndef _block_k_711__func
#define _block_k_711__func
__device__ int _block_k_711_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1064 = ((({ int _temp_var_1065 = ((({ int _temp_var_1066 = ((i % 4));
        (_temp_var_1066 == 0 ? indices.field_0 : (_temp_var_1066 == 1 ? indices.field_1 : (_temp_var_1066 == 2 ? indices.field_2 : (_temp_var_1066 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1065 == 0 ? indices.field_0 : (_temp_var_1065 == 1 ? indices.field_1 : (_temp_var_1065 == 2 ? indices.field_2 : (_temp_var_1065 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1064 == 0 ? indices.field_0 : (_temp_var_1064 == 1 ? indices.field_1 : (_temp_var_1064 == 2 ? indices.field_2 : (_temp_var_1064 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_713_ is already defined
#ifndef _block_k_713__func
#define _block_k_713__func
__device__ int _block_k_713_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1067 = ((({ int _temp_var_1068 = ((({ int _temp_var_1069 = ((i % 4));
        (_temp_var_1069 == 0 ? indices.field_0 : (_temp_var_1069 == 1 ? indices.field_1 : (_temp_var_1069 == 2 ? indices.field_2 : (_temp_var_1069 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1068 == 0 ? indices.field_0 : (_temp_var_1068 == 1 ? indices.field_1 : (_temp_var_1068 == 2 ? indices.field_2 : (_temp_var_1068 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1067 == 0 ? indices.field_0 : (_temp_var_1067 == 1 ? indices.field_1 : (_temp_var_1067 == 2 ? indices.field_2 : (_temp_var_1067 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_715_ is already defined
#ifndef _block_k_715__func
#define _block_k_715__func
__device__ int _block_k_715_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1070 = ((({ int _temp_var_1071 = ((({ int _temp_var_1072 = ((i % 4));
        (_temp_var_1072 == 0 ? indices.field_0 : (_temp_var_1072 == 1 ? indices.field_1 : (_temp_var_1072 == 2 ? indices.field_2 : (_temp_var_1072 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1071 == 0 ? indices.field_0 : (_temp_var_1071 == 1 ? indices.field_1 : (_temp_var_1071 == 2 ? indices.field_2 : (_temp_var_1071 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1070 == 0 ? indices.field_0 : (_temp_var_1070 == 1 ? indices.field_1 : (_temp_var_1070 == 2 ? indices.field_2 : (_temp_var_1070 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_717_ is already defined
#ifndef _block_k_717__func
#define _block_k_717__func
__device__ int _block_k_717_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1073 = ((({ int _temp_var_1074 = ((({ int _temp_var_1075 = ((i % 4));
        (_temp_var_1075 == 0 ? indices.field_0 : (_temp_var_1075 == 1 ? indices.field_1 : (_temp_var_1075 == 2 ? indices.field_2 : (_temp_var_1075 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1074 == 0 ? indices.field_0 : (_temp_var_1074 == 1 ? indices.field_1 : (_temp_var_1074 == 2 ? indices.field_2 : (_temp_var_1074 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1073 == 0 ? indices.field_0 : (_temp_var_1073 == 1 ? indices.field_1 : (_temp_var_1073 == 2 ? indices.field_2 : (_temp_var_1073 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_719_ is already defined
#ifndef _block_k_719__func
#define _block_k_719__func
__device__ int _block_k_719_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1076 = ((({ int _temp_var_1077 = ((({ int _temp_var_1078 = ((i % 4));
        (_temp_var_1078 == 0 ? indices.field_0 : (_temp_var_1078 == 1 ? indices.field_1 : (_temp_var_1078 == 2 ? indices.field_2 : (_temp_var_1078 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1077 == 0 ? indices.field_0 : (_temp_var_1077 == 1 ? indices.field_1 : (_temp_var_1077 == 2 ? indices.field_2 : (_temp_var_1077 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1076 == 0 ? indices.field_0 : (_temp_var_1076 == 1 ? indices.field_1 : (_temp_var_1076 == 2 ? indices.field_2 : (_temp_var_1076 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_721_ is already defined
#ifndef _block_k_721__func
#define _block_k_721__func
__device__ int _block_k_721_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1079 = ((({ int _temp_var_1080 = ((({ int _temp_var_1081 = ((i % 4));
        (_temp_var_1081 == 0 ? indices.field_0 : (_temp_var_1081 == 1 ? indices.field_1 : (_temp_var_1081 == 2 ? indices.field_2 : (_temp_var_1081 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1080 == 0 ? indices.field_0 : (_temp_var_1080 == 1 ? indices.field_1 : (_temp_var_1080 == 2 ? indices.field_2 : (_temp_var_1080 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1079 == 0 ? indices.field_0 : (_temp_var_1079 == 1 ? indices.field_1 : (_temp_var_1079 == 2 ? indices.field_2 : (_temp_var_1079 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_723_ is already defined
#ifndef _block_k_723__func
#define _block_k_723__func
__device__ int _block_k_723_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1082 = ((({ int _temp_var_1083 = ((({ int _temp_var_1084 = ((i % 4));
        (_temp_var_1084 == 0 ? indices.field_0 : (_temp_var_1084 == 1 ? indices.field_1 : (_temp_var_1084 == 2 ? indices.field_2 : (_temp_var_1084 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1083 == 0 ? indices.field_0 : (_temp_var_1083 == 1 ? indices.field_1 : (_temp_var_1083 == 2 ? indices.field_2 : (_temp_var_1083 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1082 == 0 ? indices.field_0 : (_temp_var_1082 == 1 ? indices.field_1 : (_temp_var_1082 == 2 ? indices.field_2 : (_temp_var_1082 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_725_ is already defined
#ifndef _block_k_725__func
#define _block_k_725__func
__device__ int _block_k_725_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1085 = ((({ int _temp_var_1086 = ((({ int _temp_var_1087 = ((i % 4));
        (_temp_var_1087 == 0 ? indices.field_0 : (_temp_var_1087 == 1 ? indices.field_1 : (_temp_var_1087 == 2 ? indices.field_2 : (_temp_var_1087 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1086 == 0 ? indices.field_0 : (_temp_var_1086 == 1 ? indices.field_1 : (_temp_var_1086 == 2 ? indices.field_2 : (_temp_var_1086 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1085 == 0 ? indices.field_0 : (_temp_var_1085 == 1 ? indices.field_1 : (_temp_var_1085 == 2 ? indices.field_2 : (_temp_var_1085 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_727_ is already defined
#ifndef _block_k_727__func
#define _block_k_727__func
__device__ int _block_k_727_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1088 = ((({ int _temp_var_1089 = ((({ int _temp_var_1090 = ((i % 4));
        (_temp_var_1090 == 0 ? indices.field_0 : (_temp_var_1090 == 1 ? indices.field_1 : (_temp_var_1090 == 2 ? indices.field_2 : (_temp_var_1090 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1089 == 0 ? indices.field_0 : (_temp_var_1089 == 1 ? indices.field_1 : (_temp_var_1089 == 2 ? indices.field_2 : (_temp_var_1089 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1088 == 0 ? indices.field_0 : (_temp_var_1088 == 1 ? indices.field_1 : (_temp_var_1088 == 2 ? indices.field_2 : (_temp_var_1088 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_729_ is already defined
#ifndef _block_k_729__func
#define _block_k_729__func
__device__ int _block_k_729_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1091 = ((({ int _temp_var_1092 = ((({ int _temp_var_1093 = ((i % 4));
        (_temp_var_1093 == 0 ? indices.field_0 : (_temp_var_1093 == 1 ? indices.field_1 : (_temp_var_1093 == 2 ? indices.field_2 : (_temp_var_1093 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1092 == 0 ? indices.field_0 : (_temp_var_1092 == 1 ? indices.field_1 : (_temp_var_1092 == 2 ? indices.field_2 : (_temp_var_1092 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1091 == 0 ? indices.field_0 : (_temp_var_1091 == 1 ? indices.field_1 : (_temp_var_1091 == 2 ? indices.field_2 : (_temp_var_1091 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_731_ is already defined
#ifndef _block_k_731__func
#define _block_k_731__func
__device__ int _block_k_731_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1094 = ((({ int _temp_var_1095 = ((({ int _temp_var_1096 = ((i % 4));
        (_temp_var_1096 == 0 ? indices.field_0 : (_temp_var_1096 == 1 ? indices.field_1 : (_temp_var_1096 == 2 ? indices.field_2 : (_temp_var_1096 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1095 == 0 ? indices.field_0 : (_temp_var_1095 == 1 ? indices.field_1 : (_temp_var_1095 == 2 ? indices.field_2 : (_temp_var_1095 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1094 == 0 ? indices.field_0 : (_temp_var_1094 == 1 ? indices.field_1 : (_temp_var_1094 == 2 ? indices.field_2 : (_temp_var_1094 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_733_ is already defined
#ifndef _block_k_733__func
#define _block_k_733__func
__device__ int _block_k_733_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1097 = ((({ int _temp_var_1098 = ((({ int _temp_var_1099 = ((i % 4));
        (_temp_var_1099 == 0 ? indices.field_0 : (_temp_var_1099 == 1 ? indices.field_1 : (_temp_var_1099 == 2 ? indices.field_2 : (_temp_var_1099 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1098 == 0 ? indices.field_0 : (_temp_var_1098 == 1 ? indices.field_1 : (_temp_var_1098 == 2 ? indices.field_2 : (_temp_var_1098 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1097 == 0 ? indices.field_0 : (_temp_var_1097 == 1 ? indices.field_1 : (_temp_var_1097 == 2 ? indices.field_2 : (_temp_var_1097 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_735_ is already defined
#ifndef _block_k_735__func
#define _block_k_735__func
__device__ int _block_k_735_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1100 = ((({ int _temp_var_1101 = ((({ int _temp_var_1102 = ((i % 4));
        (_temp_var_1102 == 0 ? indices.field_0 : (_temp_var_1102 == 1 ? indices.field_1 : (_temp_var_1102 == 2 ? indices.field_2 : (_temp_var_1102 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1101 == 0 ? indices.field_0 : (_temp_var_1101 == 1 ? indices.field_1 : (_temp_var_1101 == 2 ? indices.field_2 : (_temp_var_1101 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1100 == 0 ? indices.field_0 : (_temp_var_1100 == 1 ? indices.field_1 : (_temp_var_1100 == 2 ? indices.field_2 : (_temp_var_1100 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_737_ is already defined
#ifndef _block_k_737__func
#define _block_k_737__func
__device__ int _block_k_737_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1103 = ((({ int _temp_var_1104 = ((({ int _temp_var_1105 = ((i % 4));
        (_temp_var_1105 == 0 ? indices.field_0 : (_temp_var_1105 == 1 ? indices.field_1 : (_temp_var_1105 == 2 ? indices.field_2 : (_temp_var_1105 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1104 == 0 ? indices.field_0 : (_temp_var_1104 == 1 ? indices.field_1 : (_temp_var_1104 == 2 ? indices.field_2 : (_temp_var_1104 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1103 == 0 ? indices.field_0 : (_temp_var_1103 == 1 ? indices.field_1 : (_temp_var_1103 == 2 ? indices.field_2 : (_temp_var_1103 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_739_ is already defined
#ifndef _block_k_739__func
#define _block_k_739__func
__device__ int _block_k_739_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1106 = ((({ int _temp_var_1107 = ((({ int _temp_var_1108 = ((i % 4));
        (_temp_var_1108 == 0 ? indices.field_0 : (_temp_var_1108 == 1 ? indices.field_1 : (_temp_var_1108 == 2 ? indices.field_2 : (_temp_var_1108 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1107 == 0 ? indices.field_0 : (_temp_var_1107 == 1 ? indices.field_1 : (_temp_var_1107 == 2 ? indices.field_2 : (_temp_var_1107 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1106 == 0 ? indices.field_0 : (_temp_var_1106 == 1 ? indices.field_1 : (_temp_var_1106 == 2 ? indices.field_2 : (_temp_var_1106 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_741_ is already defined
#ifndef _block_k_741__func
#define _block_k_741__func
__device__ int _block_k_741_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1109 = ((({ int _temp_var_1110 = ((({ int _temp_var_1111 = ((i % 4));
        (_temp_var_1111 == 0 ? indices.field_0 : (_temp_var_1111 == 1 ? indices.field_1 : (_temp_var_1111 == 2 ? indices.field_2 : (_temp_var_1111 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1110 == 0 ? indices.field_0 : (_temp_var_1110 == 1 ? indices.field_1 : (_temp_var_1110 == 2 ? indices.field_2 : (_temp_var_1110 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1109 == 0 ? indices.field_0 : (_temp_var_1109 == 1 ? indices.field_1 : (_temp_var_1109 == 2 ? indices.field_2 : (_temp_var_1109 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_743_ is already defined
#ifndef _block_k_743__func
#define _block_k_743__func
__device__ int _block_k_743_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1112 = ((({ int _temp_var_1113 = ((({ int _temp_var_1114 = ((i % 4));
        (_temp_var_1114 == 0 ? indices.field_0 : (_temp_var_1114 == 1 ? indices.field_1 : (_temp_var_1114 == 2 ? indices.field_2 : (_temp_var_1114 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1113 == 0 ? indices.field_0 : (_temp_var_1113 == 1 ? indices.field_1 : (_temp_var_1113 == 2 ? indices.field_2 : (_temp_var_1113 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1112 == 0 ? indices.field_0 : (_temp_var_1112 == 1 ? indices.field_1 : (_temp_var_1112 == 2 ? indices.field_2 : (_temp_var_1112 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_745_ is already defined
#ifndef _block_k_745__func
#define _block_k_745__func
__device__ int _block_k_745_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1115 = ((({ int _temp_var_1116 = ((({ int _temp_var_1117 = ((i % 4));
        (_temp_var_1117 == 0 ? indices.field_0 : (_temp_var_1117 == 1 ? indices.field_1 : (_temp_var_1117 == 2 ? indices.field_2 : (_temp_var_1117 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1116 == 0 ? indices.field_0 : (_temp_var_1116 == 1 ? indices.field_1 : (_temp_var_1116 == 2 ? indices.field_2 : (_temp_var_1116 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1115 == 0 ? indices.field_0 : (_temp_var_1115 == 1 ? indices.field_1 : (_temp_var_1115 == 2 ? indices.field_2 : (_temp_var_1115 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_747_ is already defined
#ifndef _block_k_747__func
#define _block_k_747__func
__device__ int _block_k_747_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1118 = ((({ int _temp_var_1119 = ((({ int _temp_var_1120 = ((i % 4));
        (_temp_var_1120 == 0 ? indices.field_0 : (_temp_var_1120 == 1 ? indices.field_1 : (_temp_var_1120 == 2 ? indices.field_2 : (_temp_var_1120 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1119 == 0 ? indices.field_0 : (_temp_var_1119 == 1 ? indices.field_1 : (_temp_var_1119 == 2 ? indices.field_2 : (_temp_var_1119 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1118 == 0 ? indices.field_0 : (_temp_var_1118 == 1 ? indices.field_1 : (_temp_var_1118 == 2 ? indices.field_2 : (_temp_var_1118 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_749_ is already defined
#ifndef _block_k_749__func
#define _block_k_749__func
__device__ int _block_k_749_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1121 = ((({ int _temp_var_1122 = ((({ int _temp_var_1123 = ((i % 4));
        (_temp_var_1123 == 0 ? indices.field_0 : (_temp_var_1123 == 1 ? indices.field_1 : (_temp_var_1123 == 2 ? indices.field_2 : (_temp_var_1123 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1122 == 0 ? indices.field_0 : (_temp_var_1122 == 1 ? indices.field_1 : (_temp_var_1122 == 2 ? indices.field_2 : (_temp_var_1122 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1121 == 0 ? indices.field_0 : (_temp_var_1121 == 1 ? indices.field_1 : (_temp_var_1121 == 2 ? indices.field_2 : (_temp_var_1121 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_751_ is already defined
#ifndef _block_k_751__func
#define _block_k_751__func
__device__ int _block_k_751_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1124 = ((({ int _temp_var_1125 = ((({ int _temp_var_1126 = ((i % 4));
        (_temp_var_1126 == 0 ? indices.field_0 : (_temp_var_1126 == 1 ? indices.field_1 : (_temp_var_1126 == 2 ? indices.field_2 : (_temp_var_1126 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1125 == 0 ? indices.field_0 : (_temp_var_1125 == 1 ? indices.field_1 : (_temp_var_1125 == 2 ? indices.field_2 : (_temp_var_1125 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1124 == 0 ? indices.field_0 : (_temp_var_1124 == 1 ? indices.field_1 : (_temp_var_1124 == 2 ? indices.field_2 : (_temp_var_1124 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_753_ is already defined
#ifndef _block_k_753__func
#define _block_k_753__func
__device__ int _block_k_753_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1127 = ((({ int _temp_var_1128 = ((({ int _temp_var_1129 = ((i % 4));
        (_temp_var_1129 == 0 ? indices.field_0 : (_temp_var_1129 == 1 ? indices.field_1 : (_temp_var_1129 == 2 ? indices.field_2 : (_temp_var_1129 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1128 == 0 ? indices.field_0 : (_temp_var_1128 == 1 ? indices.field_1 : (_temp_var_1128 == 2 ? indices.field_2 : (_temp_var_1128 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1127 == 0 ? indices.field_0 : (_temp_var_1127 == 1 ? indices.field_1 : (_temp_var_1127 == 2 ? indices.field_2 : (_temp_var_1127 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_755_ is already defined
#ifndef _block_k_755__func
#define _block_k_755__func
__device__ int _block_k_755_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1130 = ((({ int _temp_var_1131 = ((({ int _temp_var_1132 = ((i % 4));
        (_temp_var_1132 == 0 ? indices.field_0 : (_temp_var_1132 == 1 ? indices.field_1 : (_temp_var_1132 == 2 ? indices.field_2 : (_temp_var_1132 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1131 == 0 ? indices.field_0 : (_temp_var_1131 == 1 ? indices.field_1 : (_temp_var_1131 == 2 ? indices.field_2 : (_temp_var_1131 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1130 == 0 ? indices.field_0 : (_temp_var_1130 == 1 ? indices.field_1 : (_temp_var_1130 == 2 ? indices.field_2 : (_temp_var_1130 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_757_ is already defined
#ifndef _block_k_757__func
#define _block_k_757__func
__device__ int _block_k_757_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1133 = ((({ int _temp_var_1134 = ((({ int _temp_var_1135 = ((i % 4));
        (_temp_var_1135 == 0 ? indices.field_0 : (_temp_var_1135 == 1 ? indices.field_1 : (_temp_var_1135 == 2 ? indices.field_2 : (_temp_var_1135 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1134 == 0 ? indices.field_0 : (_temp_var_1134 == 1 ? indices.field_1 : (_temp_var_1134 == 2 ? indices.field_2 : (_temp_var_1134 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1133 == 0 ? indices.field_0 : (_temp_var_1133 == 1 ? indices.field_1 : (_temp_var_1133 == 2 ? indices.field_2 : (_temp_var_1133 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_759_ is already defined
#ifndef _block_k_759__func
#define _block_k_759__func
__device__ int _block_k_759_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1136 = ((({ int _temp_var_1137 = ((({ int _temp_var_1138 = ((i % 4));
        (_temp_var_1138 == 0 ? indices.field_0 : (_temp_var_1138 == 1 ? indices.field_1 : (_temp_var_1138 == 2 ? indices.field_2 : (_temp_var_1138 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1137 == 0 ? indices.field_0 : (_temp_var_1137 == 1 ? indices.field_1 : (_temp_var_1137 == 2 ? indices.field_2 : (_temp_var_1137 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1136 == 0 ? indices.field_0 : (_temp_var_1136 == 1 ? indices.field_1 : (_temp_var_1136 == 2 ? indices.field_2 : (_temp_var_1136 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_761_ is already defined
#ifndef _block_k_761__func
#define _block_k_761__func
__device__ int _block_k_761_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1139 = ((({ int _temp_var_1140 = ((({ int _temp_var_1141 = ((i % 4));
        (_temp_var_1141 == 0 ? indices.field_0 : (_temp_var_1141 == 1 ? indices.field_1 : (_temp_var_1141 == 2 ? indices.field_2 : (_temp_var_1141 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1140 == 0 ? indices.field_0 : (_temp_var_1140 == 1 ? indices.field_1 : (_temp_var_1140 == 2 ? indices.field_2 : (_temp_var_1140 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1139 == 0 ? indices.field_0 : (_temp_var_1139 == 1 ? indices.field_1 : (_temp_var_1139 == 2 ? indices.field_2 : (_temp_var_1139 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_763_ is already defined
#ifndef _block_k_763__func
#define _block_k_763__func
__device__ int _block_k_763_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1142 = ((({ int _temp_var_1143 = ((({ int _temp_var_1144 = ((i % 4));
        (_temp_var_1144 == 0 ? indices.field_0 : (_temp_var_1144 == 1 ? indices.field_1 : (_temp_var_1144 == 2 ? indices.field_2 : (_temp_var_1144 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1143 == 0 ? indices.field_0 : (_temp_var_1143 == 1 ? indices.field_1 : (_temp_var_1143 == 2 ? indices.field_2 : (_temp_var_1143 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1142 == 0 ? indices.field_0 : (_temp_var_1142 == 1 ? indices.field_1 : (_temp_var_1142 == 2 ? indices.field_2 : (_temp_var_1142 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_765_ is already defined
#ifndef _block_k_765__func
#define _block_k_765__func
__device__ int _block_k_765_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1145 = ((({ int _temp_var_1146 = ((({ int _temp_var_1147 = ((i % 4));
        (_temp_var_1147 == 0 ? indices.field_0 : (_temp_var_1147 == 1 ? indices.field_1 : (_temp_var_1147 == 2 ? indices.field_2 : (_temp_var_1147 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1146 == 0 ? indices.field_0 : (_temp_var_1146 == 1 ? indices.field_1 : (_temp_var_1146 == 2 ? indices.field_2 : (_temp_var_1146 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1145 == 0 ? indices.field_0 : (_temp_var_1145 == 1 ? indices.field_1 : (_temp_var_1145 == 2 ? indices.field_2 : (_temp_var_1145 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_767_ is already defined
#ifndef _block_k_767__func
#define _block_k_767__func
__device__ int _block_k_767_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1148 = ((({ int _temp_var_1149 = ((({ int _temp_var_1150 = ((i % 4));
        (_temp_var_1150 == 0 ? indices.field_0 : (_temp_var_1150 == 1 ? indices.field_1 : (_temp_var_1150 == 2 ? indices.field_2 : (_temp_var_1150 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1149 == 0 ? indices.field_0 : (_temp_var_1149 == 1 ? indices.field_1 : (_temp_var_1149 == 2 ? indices.field_2 : (_temp_var_1149 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1148 == 0 ? indices.field_0 : (_temp_var_1148 == 1 ? indices.field_1 : (_temp_var_1148 == 2 ? indices.field_2 : (_temp_var_1148 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_769_ is already defined
#ifndef _block_k_769__func
#define _block_k_769__func
__device__ int _block_k_769_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1151 = ((({ int _temp_var_1152 = ((({ int _temp_var_1153 = ((i % 4));
        (_temp_var_1153 == 0 ? indices.field_0 : (_temp_var_1153 == 1 ? indices.field_1 : (_temp_var_1153 == 2 ? indices.field_2 : (_temp_var_1153 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1152 == 0 ? indices.field_0 : (_temp_var_1152 == 1 ? indices.field_1 : (_temp_var_1152 == 2 ? indices.field_2 : (_temp_var_1152 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1151 == 0 ? indices.field_0 : (_temp_var_1151 == 1 ? indices.field_1 : (_temp_var_1151 == 2 ? indices.field_2 : (_temp_var_1151 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_771_ is already defined
#ifndef _block_k_771__func
#define _block_k_771__func
__device__ int _block_k_771_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1154 = ((({ int _temp_var_1155 = ((({ int _temp_var_1156 = ((i % 4));
        (_temp_var_1156 == 0 ? indices.field_0 : (_temp_var_1156 == 1 ? indices.field_1 : (_temp_var_1156 == 2 ? indices.field_2 : (_temp_var_1156 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1155 == 0 ? indices.field_0 : (_temp_var_1155 == 1 ? indices.field_1 : (_temp_var_1155 == 2 ? indices.field_2 : (_temp_var_1155 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1154 == 0 ? indices.field_0 : (_temp_var_1154 == 1 ? indices.field_1 : (_temp_var_1154 == 2 ? indices.field_2 : (_temp_var_1154 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_773_ is already defined
#ifndef _block_k_773__func
#define _block_k_773__func
__device__ int _block_k_773_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1157 = ((({ int _temp_var_1158 = ((({ int _temp_var_1159 = ((i % 4));
        (_temp_var_1159 == 0 ? indices.field_0 : (_temp_var_1159 == 1 ? indices.field_1 : (_temp_var_1159 == 2 ? indices.field_2 : (_temp_var_1159 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1158 == 0 ? indices.field_0 : (_temp_var_1158 == 1 ? indices.field_1 : (_temp_var_1158 == 2 ? indices.field_2 : (_temp_var_1158 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1157 == 0 ? indices.field_0 : (_temp_var_1157 == 1 ? indices.field_1 : (_temp_var_1157 == 2 ? indices.field_2 : (_temp_var_1157 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_775_ is already defined
#ifndef _block_k_775__func
#define _block_k_775__func
__device__ int _block_k_775_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1160 = ((({ int _temp_var_1161 = ((({ int _temp_var_1162 = ((i % 4));
        (_temp_var_1162 == 0 ? indices.field_0 : (_temp_var_1162 == 1 ? indices.field_1 : (_temp_var_1162 == 2 ? indices.field_2 : (_temp_var_1162 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1161 == 0 ? indices.field_0 : (_temp_var_1161 == 1 ? indices.field_1 : (_temp_var_1161 == 2 ? indices.field_2 : (_temp_var_1161 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1160 == 0 ? indices.field_0 : (_temp_var_1160 == 1 ? indices.field_1 : (_temp_var_1160 == 2 ? indices.field_2 : (_temp_var_1160 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_777_ is already defined
#ifndef _block_k_777__func
#define _block_k_777__func
__device__ int _block_k_777_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1163 = ((({ int _temp_var_1164 = ((({ int _temp_var_1165 = ((i % 4));
        (_temp_var_1165 == 0 ? indices.field_0 : (_temp_var_1165 == 1 ? indices.field_1 : (_temp_var_1165 == 2 ? indices.field_2 : (_temp_var_1165 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1164 == 0 ? indices.field_0 : (_temp_var_1164 == 1 ? indices.field_1 : (_temp_var_1164 == 2 ? indices.field_2 : (_temp_var_1164 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1163 == 0 ? indices.field_0 : (_temp_var_1163 == 1 ? indices.field_1 : (_temp_var_1163 == 2 ? indices.field_2 : (_temp_var_1163 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_779_ is already defined
#ifndef _block_k_779__func
#define _block_k_779__func
__device__ int _block_k_779_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1166 = ((({ int _temp_var_1167 = ((({ int _temp_var_1168 = ((i % 4));
        (_temp_var_1168 == 0 ? indices.field_0 : (_temp_var_1168 == 1 ? indices.field_1 : (_temp_var_1168 == 2 ? indices.field_2 : (_temp_var_1168 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1167 == 0 ? indices.field_0 : (_temp_var_1167 == 1 ? indices.field_1 : (_temp_var_1167 == 2 ? indices.field_2 : (_temp_var_1167 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1166 == 0 ? indices.field_0 : (_temp_var_1166 == 1 ? indices.field_1 : (_temp_var_1166 == 2 ? indices.field_2 : (_temp_var_1166 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_781_ is already defined
#ifndef _block_k_781__func
#define _block_k_781__func
__device__ int _block_k_781_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1169 = ((({ int _temp_var_1170 = ((({ int _temp_var_1171 = ((i % 4));
        (_temp_var_1171 == 0 ? indices.field_0 : (_temp_var_1171 == 1 ? indices.field_1 : (_temp_var_1171 == 2 ? indices.field_2 : (_temp_var_1171 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1170 == 0 ? indices.field_0 : (_temp_var_1170 == 1 ? indices.field_1 : (_temp_var_1170 == 2 ? indices.field_2 : (_temp_var_1170 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1169 == 0 ? indices.field_0 : (_temp_var_1169 == 1 ? indices.field_1 : (_temp_var_1169 == 2 ? indices.field_2 : (_temp_var_1169 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_783_ is already defined
#ifndef _block_k_783__func
#define _block_k_783__func
__device__ int _block_k_783_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1172 = ((({ int _temp_var_1173 = ((({ int _temp_var_1174 = ((i % 4));
        (_temp_var_1174 == 0 ? indices.field_0 : (_temp_var_1174 == 1 ? indices.field_1 : (_temp_var_1174 == 2 ? indices.field_2 : (_temp_var_1174 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1173 == 0 ? indices.field_0 : (_temp_var_1173 == 1 ? indices.field_1 : (_temp_var_1173 == 2 ? indices.field_2 : (_temp_var_1173 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1172 == 0 ? indices.field_0 : (_temp_var_1172 == 1 ? indices.field_1 : (_temp_var_1172 == 2 ? indices.field_2 : (_temp_var_1172 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_785_ is already defined
#ifndef _block_k_785__func
#define _block_k_785__func
__device__ int _block_k_785_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1175 = ((({ int _temp_var_1176 = ((({ int _temp_var_1177 = ((i % 4));
        (_temp_var_1177 == 0 ? indices.field_0 : (_temp_var_1177 == 1 ? indices.field_1 : (_temp_var_1177 == 2 ? indices.field_2 : (_temp_var_1177 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1176 == 0 ? indices.field_0 : (_temp_var_1176 == 1 ? indices.field_1 : (_temp_var_1176 == 2 ? indices.field_2 : (_temp_var_1176 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1175 == 0 ? indices.field_0 : (_temp_var_1175 == 1 ? indices.field_1 : (_temp_var_1175 == 2 ? indices.field_2 : (_temp_var_1175 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_787_ is already defined
#ifndef _block_k_787__func
#define _block_k_787__func
__device__ int _block_k_787_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1178 = ((({ int _temp_var_1179 = ((({ int _temp_var_1180 = ((i % 4));
        (_temp_var_1180 == 0 ? indices.field_0 : (_temp_var_1180 == 1 ? indices.field_1 : (_temp_var_1180 == 2 ? indices.field_2 : (_temp_var_1180 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1179 == 0 ? indices.field_0 : (_temp_var_1179 == 1 ? indices.field_1 : (_temp_var_1179 == 2 ? indices.field_2 : (_temp_var_1179 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1178 == 0 ? indices.field_0 : (_temp_var_1178 == 1 ? indices.field_1 : (_temp_var_1178 == 2 ? indices.field_2 : (_temp_var_1178 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_789_ is already defined
#ifndef _block_k_789__func
#define _block_k_789__func
__device__ int _block_k_789_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1181 = ((({ int _temp_var_1182 = ((({ int _temp_var_1183 = ((i % 4));
        (_temp_var_1183 == 0 ? indices.field_0 : (_temp_var_1183 == 1 ? indices.field_1 : (_temp_var_1183 == 2 ? indices.field_2 : (_temp_var_1183 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1182 == 0 ? indices.field_0 : (_temp_var_1182 == 1 ? indices.field_1 : (_temp_var_1182 == 2 ? indices.field_2 : (_temp_var_1182 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1181 == 0 ? indices.field_0 : (_temp_var_1181 == 1 ? indices.field_1 : (_temp_var_1181 == 2 ? indices.field_2 : (_temp_var_1181 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_791_ is already defined
#ifndef _block_k_791__func
#define _block_k_791__func
__device__ int _block_k_791_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1184 = ((({ int _temp_var_1185 = ((({ int _temp_var_1186 = ((i % 4));
        (_temp_var_1186 == 0 ? indices.field_0 : (_temp_var_1186 == 1 ? indices.field_1 : (_temp_var_1186 == 2 ? indices.field_2 : (_temp_var_1186 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1185 == 0 ? indices.field_0 : (_temp_var_1185 == 1 ? indices.field_1 : (_temp_var_1185 == 2 ? indices.field_2 : (_temp_var_1185 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1184 == 0 ? indices.field_0 : (_temp_var_1184 == 1 ? indices.field_1 : (_temp_var_1184 == 2 ? indices.field_2 : (_temp_var_1184 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_793_ is already defined
#ifndef _block_k_793__func
#define _block_k_793__func
__device__ int _block_k_793_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1187 = ((({ int _temp_var_1188 = ((({ int _temp_var_1189 = ((i % 4));
        (_temp_var_1189 == 0 ? indices.field_0 : (_temp_var_1189 == 1 ? indices.field_1 : (_temp_var_1189 == 2 ? indices.field_2 : (_temp_var_1189 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1188 == 0 ? indices.field_0 : (_temp_var_1188 == 1 ? indices.field_1 : (_temp_var_1188 == 2 ? indices.field_2 : (_temp_var_1188 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1187 == 0 ? indices.field_0 : (_temp_var_1187 == 1 ? indices.field_1 : (_temp_var_1187 == 2 ? indices.field_2 : (_temp_var_1187 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_795_ is already defined
#ifndef _block_k_795__func
#define _block_k_795__func
__device__ int _block_k_795_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1190 = ((({ int _temp_var_1191 = ((({ int _temp_var_1192 = ((i % 4));
        (_temp_var_1192 == 0 ? indices.field_0 : (_temp_var_1192 == 1 ? indices.field_1 : (_temp_var_1192 == 2 ? indices.field_2 : (_temp_var_1192 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1191 == 0 ? indices.field_0 : (_temp_var_1191 == 1 ? indices.field_1 : (_temp_var_1191 == 2 ? indices.field_2 : (_temp_var_1191 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1190 == 0 ? indices.field_0 : (_temp_var_1190 == 1 ? indices.field_1 : (_temp_var_1190 == 2 ? indices.field_2 : (_temp_var_1190 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_797_ is already defined
#ifndef _block_k_797__func
#define _block_k_797__func
__device__ int _block_k_797_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1193 = ((({ int _temp_var_1194 = ((({ int _temp_var_1195 = ((i % 4));
        (_temp_var_1195 == 0 ? indices.field_0 : (_temp_var_1195 == 1 ? indices.field_1 : (_temp_var_1195 == 2 ? indices.field_2 : (_temp_var_1195 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1194 == 0 ? indices.field_0 : (_temp_var_1194 == 1 ? indices.field_1 : (_temp_var_1194 == 2 ? indices.field_2 : (_temp_var_1194 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1193 == 0 ? indices.field_0 : (_temp_var_1193 == 1 ? indices.field_1 : (_temp_var_1193 == 2 ? indices.field_2 : (_temp_var_1193 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_799_ is already defined
#ifndef _block_k_799__func
#define _block_k_799__func
__device__ int _block_k_799_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1196 = ((({ int _temp_var_1197 = ((({ int _temp_var_1198 = ((i % 4));
        (_temp_var_1198 == 0 ? indices.field_0 : (_temp_var_1198 == 1 ? indices.field_1 : (_temp_var_1198 == 2 ? indices.field_2 : (_temp_var_1198 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1197 == 0 ? indices.field_0 : (_temp_var_1197 == 1 ? indices.field_1 : (_temp_var_1197 == 2 ? indices.field_2 : (_temp_var_1197 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1196 == 0 ? indices.field_0 : (_temp_var_1196 == 1 ? indices.field_1 : (_temp_var_1196 == 2 ? indices.field_2 : (_temp_var_1196 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_801_ is already defined
#ifndef _block_k_801__func
#define _block_k_801__func
__device__ int _block_k_801_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1199 = ((({ int _temp_var_1200 = ((({ int _temp_var_1201 = ((i % 4));
        (_temp_var_1201 == 0 ? indices.field_0 : (_temp_var_1201 == 1 ? indices.field_1 : (_temp_var_1201 == 2 ? indices.field_2 : (_temp_var_1201 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1200 == 0 ? indices.field_0 : (_temp_var_1200 == 1 ? indices.field_1 : (_temp_var_1200 == 2 ? indices.field_2 : (_temp_var_1200 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1199 == 0 ? indices.field_0 : (_temp_var_1199 == 1 ? indices.field_1 : (_temp_var_1199 == 2 ? indices.field_2 : (_temp_var_1199 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_803_ is already defined
#ifndef _block_k_803__func
#define _block_k_803__func
__device__ int _block_k_803_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1202 = ((({ int _temp_var_1203 = ((({ int _temp_var_1204 = ((i % 4));
        (_temp_var_1204 == 0 ? indices.field_0 : (_temp_var_1204 == 1 ? indices.field_1 : (_temp_var_1204 == 2 ? indices.field_2 : (_temp_var_1204 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1203 == 0 ? indices.field_0 : (_temp_var_1203 == 1 ? indices.field_1 : (_temp_var_1203 == 2 ? indices.field_2 : (_temp_var_1203 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1202 == 0 ? indices.field_0 : (_temp_var_1202 == 1 ? indices.field_1 : (_temp_var_1202 == 2 ? indices.field_2 : (_temp_var_1202 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_805_ is already defined
#ifndef _block_k_805__func
#define _block_k_805__func
__device__ int _block_k_805_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1205 = ((({ int _temp_var_1206 = ((({ int _temp_var_1207 = ((i % 4));
        (_temp_var_1207 == 0 ? indices.field_0 : (_temp_var_1207 == 1 ? indices.field_1 : (_temp_var_1207 == 2 ? indices.field_2 : (_temp_var_1207 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1206 == 0 ? indices.field_0 : (_temp_var_1206 == 1 ? indices.field_1 : (_temp_var_1206 == 2 ? indices.field_2 : (_temp_var_1206 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1205 == 0 ? indices.field_0 : (_temp_var_1205 == 1 ? indices.field_1 : (_temp_var_1205 == 2 ? indices.field_2 : (_temp_var_1205 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_807_ is already defined
#ifndef _block_k_807__func
#define _block_k_807__func
__device__ int _block_k_807_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1208 = ((({ int _temp_var_1209 = ((({ int _temp_var_1210 = ((i % 4));
        (_temp_var_1210 == 0 ? indices.field_0 : (_temp_var_1210 == 1 ? indices.field_1 : (_temp_var_1210 == 2 ? indices.field_2 : (_temp_var_1210 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1209 == 0 ? indices.field_0 : (_temp_var_1209 == 1 ? indices.field_1 : (_temp_var_1209 == 2 ? indices.field_2 : (_temp_var_1209 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1208 == 0 ? indices.field_0 : (_temp_var_1208 == 1 ? indices.field_1 : (_temp_var_1208 == 2 ? indices.field_2 : (_temp_var_1208 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_809_ is already defined
#ifndef _block_k_809__func
#define _block_k_809__func
__device__ int _block_k_809_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1211 = ((({ int _temp_var_1212 = ((({ int _temp_var_1213 = ((i % 4));
        (_temp_var_1213 == 0 ? indices.field_0 : (_temp_var_1213 == 1 ? indices.field_1 : (_temp_var_1213 == 2 ? indices.field_2 : (_temp_var_1213 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1212 == 0 ? indices.field_0 : (_temp_var_1212 == 1 ? indices.field_1 : (_temp_var_1212 == 2 ? indices.field_2 : (_temp_var_1212 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1211 == 0 ? indices.field_0 : (_temp_var_1211 == 1 ? indices.field_1 : (_temp_var_1211 == 2 ? indices.field_2 : (_temp_var_1211 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_811_ is already defined
#ifndef _block_k_811__func
#define _block_k_811__func
__device__ int _block_k_811_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1214 = ((({ int _temp_var_1215 = ((({ int _temp_var_1216 = ((i % 4));
        (_temp_var_1216 == 0 ? indices.field_0 : (_temp_var_1216 == 1 ? indices.field_1 : (_temp_var_1216 == 2 ? indices.field_2 : (_temp_var_1216 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1215 == 0 ? indices.field_0 : (_temp_var_1215 == 1 ? indices.field_1 : (_temp_var_1215 == 2 ? indices.field_2 : (_temp_var_1215 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1214 == 0 ? indices.field_0 : (_temp_var_1214 == 1 ? indices.field_1 : (_temp_var_1214 == 2 ? indices.field_2 : (_temp_var_1214 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_813_ is already defined
#ifndef _block_k_813__func
#define _block_k_813__func
__device__ int _block_k_813_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1217 = ((({ int _temp_var_1218 = ((({ int _temp_var_1219 = ((i % 4));
        (_temp_var_1219 == 0 ? indices.field_0 : (_temp_var_1219 == 1 ? indices.field_1 : (_temp_var_1219 == 2 ? indices.field_2 : (_temp_var_1219 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1218 == 0 ? indices.field_0 : (_temp_var_1218 == 1 ? indices.field_1 : (_temp_var_1218 == 2 ? indices.field_2 : (_temp_var_1218 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1217 == 0 ? indices.field_0 : (_temp_var_1217 == 1 ? indices.field_1 : (_temp_var_1217 == 2 ? indices.field_2 : (_temp_var_1217 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_815_ is already defined
#ifndef _block_k_815__func
#define _block_k_815__func
__device__ int _block_k_815_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1220 = ((({ int _temp_var_1221 = ((({ int _temp_var_1222 = ((i % 4));
        (_temp_var_1222 == 0 ? indices.field_0 : (_temp_var_1222 == 1 ? indices.field_1 : (_temp_var_1222 == 2 ? indices.field_2 : (_temp_var_1222 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1221 == 0 ? indices.field_0 : (_temp_var_1221 == 1 ? indices.field_1 : (_temp_var_1221 == 2 ? indices.field_2 : (_temp_var_1221 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1220 == 0 ? indices.field_0 : (_temp_var_1220 == 1 ? indices.field_1 : (_temp_var_1220 == 2 ? indices.field_2 : (_temp_var_1220 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_817_ is already defined
#ifndef _block_k_817__func
#define _block_k_817__func
__device__ int _block_k_817_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1223 = ((({ int _temp_var_1224 = ((({ int _temp_var_1225 = ((i % 4));
        (_temp_var_1225 == 0 ? indices.field_0 : (_temp_var_1225 == 1 ? indices.field_1 : (_temp_var_1225 == 2 ? indices.field_2 : (_temp_var_1225 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1224 == 0 ? indices.field_0 : (_temp_var_1224 == 1 ? indices.field_1 : (_temp_var_1224 == 2 ? indices.field_2 : (_temp_var_1224 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1223 == 0 ? indices.field_0 : (_temp_var_1223 == 1 ? indices.field_1 : (_temp_var_1223 == 2 ? indices.field_2 : (_temp_var_1223 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_819_ is already defined
#ifndef _block_k_819__func
#define _block_k_819__func
__device__ int _block_k_819_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1226 = ((({ int _temp_var_1227 = ((({ int _temp_var_1228 = ((i % 4));
        (_temp_var_1228 == 0 ? indices.field_0 : (_temp_var_1228 == 1 ? indices.field_1 : (_temp_var_1228 == 2 ? indices.field_2 : (_temp_var_1228 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1227 == 0 ? indices.field_0 : (_temp_var_1227 == 1 ? indices.field_1 : (_temp_var_1227 == 2 ? indices.field_2 : (_temp_var_1227 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1226 == 0 ? indices.field_0 : (_temp_var_1226 == 1 ? indices.field_1 : (_temp_var_1226 == 2 ? indices.field_2 : (_temp_var_1226 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_821_ is already defined
#ifndef _block_k_821__func
#define _block_k_821__func
__device__ int _block_k_821_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1229 = ((({ int _temp_var_1230 = ((({ int _temp_var_1231 = ((i % 4));
        (_temp_var_1231 == 0 ? indices.field_0 : (_temp_var_1231 == 1 ? indices.field_1 : (_temp_var_1231 == 2 ? indices.field_2 : (_temp_var_1231 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1230 == 0 ? indices.field_0 : (_temp_var_1230 == 1 ? indices.field_1 : (_temp_var_1230 == 2 ? indices.field_2 : (_temp_var_1230 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1229 == 0 ? indices.field_0 : (_temp_var_1229 == 1 ? indices.field_1 : (_temp_var_1229 == 2 ? indices.field_2 : (_temp_var_1229 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_823_ is already defined
#ifndef _block_k_823__func
#define _block_k_823__func
__device__ int _block_k_823_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1232 = ((({ int _temp_var_1233 = ((({ int _temp_var_1234 = ((i % 4));
        (_temp_var_1234 == 0 ? indices.field_0 : (_temp_var_1234 == 1 ? indices.field_1 : (_temp_var_1234 == 2 ? indices.field_2 : (_temp_var_1234 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1233 == 0 ? indices.field_0 : (_temp_var_1233 == 1 ? indices.field_1 : (_temp_var_1233 == 2 ? indices.field_2 : (_temp_var_1233 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1232 == 0 ? indices.field_0 : (_temp_var_1232 == 1 ? indices.field_1 : (_temp_var_1232 == 2 ? indices.field_2 : (_temp_var_1232 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_825_ is already defined
#ifndef _block_k_825__func
#define _block_k_825__func
__device__ int _block_k_825_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1235 = ((({ int _temp_var_1236 = ((({ int _temp_var_1237 = ((i % 4));
        (_temp_var_1237 == 0 ? indices.field_0 : (_temp_var_1237 == 1 ? indices.field_1 : (_temp_var_1237 == 2 ? indices.field_2 : (_temp_var_1237 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1236 == 0 ? indices.field_0 : (_temp_var_1236 == 1 ? indices.field_1 : (_temp_var_1236 == 2 ? indices.field_2 : (_temp_var_1236 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1235 == 0 ? indices.field_0 : (_temp_var_1235 == 1 ? indices.field_1 : (_temp_var_1235 == 2 ? indices.field_2 : (_temp_var_1235 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_827_ is already defined
#ifndef _block_k_827__func
#define _block_k_827__func
__device__ int _block_k_827_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1238 = ((({ int _temp_var_1239 = ((({ int _temp_var_1240 = ((i % 4));
        (_temp_var_1240 == 0 ? indices.field_0 : (_temp_var_1240 == 1 ? indices.field_1 : (_temp_var_1240 == 2 ? indices.field_2 : (_temp_var_1240 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1239 == 0 ? indices.field_0 : (_temp_var_1239 == 1 ? indices.field_1 : (_temp_var_1239 == 2 ? indices.field_2 : (_temp_var_1239 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1238 == 0 ? indices.field_0 : (_temp_var_1238 == 1 ? indices.field_1 : (_temp_var_1238 == 2 ? indices.field_2 : (_temp_var_1238 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_829_ is already defined
#ifndef _block_k_829__func
#define _block_k_829__func
__device__ int _block_k_829_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1241 = ((({ int _temp_var_1242 = ((({ int _temp_var_1243 = ((i % 4));
        (_temp_var_1243 == 0 ? indices.field_0 : (_temp_var_1243 == 1 ? indices.field_1 : (_temp_var_1243 == 2 ? indices.field_2 : (_temp_var_1243 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1242 == 0 ? indices.field_0 : (_temp_var_1242 == 1 ? indices.field_1 : (_temp_var_1242 == 2 ? indices.field_2 : (_temp_var_1242 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1241 == 0 ? indices.field_0 : (_temp_var_1241 == 1 ? indices.field_1 : (_temp_var_1241 == 2 ? indices.field_2 : (_temp_var_1241 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_831_ is already defined
#ifndef _block_k_831__func
#define _block_k_831__func
__device__ int _block_k_831_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1244 = ((({ int _temp_var_1245 = ((({ int _temp_var_1246 = ((i % 4));
        (_temp_var_1246 == 0 ? indices.field_0 : (_temp_var_1246 == 1 ? indices.field_1 : (_temp_var_1246 == 2 ? indices.field_2 : (_temp_var_1246 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1245 == 0 ? indices.field_0 : (_temp_var_1245 == 1 ? indices.field_1 : (_temp_var_1245 == 2 ? indices.field_2 : (_temp_var_1245 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1244 == 0 ? indices.field_0 : (_temp_var_1244 == 1 ? indices.field_1 : (_temp_var_1244 == 2 ? indices.field_2 : (_temp_var_1244 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_833_ is already defined
#ifndef _block_k_833__func
#define _block_k_833__func
__device__ int _block_k_833_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1247 = ((({ int _temp_var_1248 = ((({ int _temp_var_1249 = ((i % 4));
        (_temp_var_1249 == 0 ? indices.field_0 : (_temp_var_1249 == 1 ? indices.field_1 : (_temp_var_1249 == 2 ? indices.field_2 : (_temp_var_1249 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1248 == 0 ? indices.field_0 : (_temp_var_1248 == 1 ? indices.field_1 : (_temp_var_1248 == 2 ? indices.field_2 : (_temp_var_1248 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1247 == 0 ? indices.field_0 : (_temp_var_1247 == 1 ? indices.field_1 : (_temp_var_1247 == 2 ? indices.field_2 : (_temp_var_1247 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_835_ is already defined
#ifndef _block_k_835__func
#define _block_k_835__func
__device__ int _block_k_835_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1250 = ((({ int _temp_var_1251 = ((({ int _temp_var_1252 = ((i % 4));
        (_temp_var_1252 == 0 ? indices.field_0 : (_temp_var_1252 == 1 ? indices.field_1 : (_temp_var_1252 == 2 ? indices.field_2 : (_temp_var_1252 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1251 == 0 ? indices.field_0 : (_temp_var_1251 == 1 ? indices.field_1 : (_temp_var_1251 == 2 ? indices.field_2 : (_temp_var_1251 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1250 == 0 ? indices.field_0 : (_temp_var_1250 == 1 ? indices.field_1 : (_temp_var_1250 == 2 ? indices.field_2 : (_temp_var_1250 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_837_ is already defined
#ifndef _block_k_837__func
#define _block_k_837__func
__device__ int _block_k_837_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1253 = ((({ int _temp_var_1254 = ((({ int _temp_var_1255 = ((i % 4));
        (_temp_var_1255 == 0 ? indices.field_0 : (_temp_var_1255 == 1 ? indices.field_1 : (_temp_var_1255 == 2 ? indices.field_2 : (_temp_var_1255 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1254 == 0 ? indices.field_0 : (_temp_var_1254 == 1 ? indices.field_1 : (_temp_var_1254 == 2 ? indices.field_2 : (_temp_var_1254 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1253 == 0 ? indices.field_0 : (_temp_var_1253 == 1 ? indices.field_1 : (_temp_var_1253 == 2 ? indices.field_2 : (_temp_var_1253 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_839_ is already defined
#ifndef _block_k_839__func
#define _block_k_839__func
__device__ int _block_k_839_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1256 = ((({ int _temp_var_1257 = ((({ int _temp_var_1258 = ((i % 4));
        (_temp_var_1258 == 0 ? indices.field_0 : (_temp_var_1258 == 1 ? indices.field_1 : (_temp_var_1258 == 2 ? indices.field_2 : (_temp_var_1258 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1257 == 0 ? indices.field_0 : (_temp_var_1257 == 1 ? indices.field_1 : (_temp_var_1257 == 2 ? indices.field_2 : (_temp_var_1257 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1256 == 0 ? indices.field_0 : (_temp_var_1256 == 1 ? indices.field_1 : (_temp_var_1256 == 2 ? indices.field_2 : (_temp_var_1256 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_841_ is already defined
#ifndef _block_k_841__func
#define _block_k_841__func
__device__ int _block_k_841_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1259 = ((({ int _temp_var_1260 = ((({ int _temp_var_1261 = ((i % 4));
        (_temp_var_1261 == 0 ? indices.field_0 : (_temp_var_1261 == 1 ? indices.field_1 : (_temp_var_1261 == 2 ? indices.field_2 : (_temp_var_1261 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1260 == 0 ? indices.field_0 : (_temp_var_1260 == 1 ? indices.field_1 : (_temp_var_1260 == 2 ? indices.field_2 : (_temp_var_1260 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1259 == 0 ? indices.field_0 : (_temp_var_1259 == 1 ? indices.field_1 : (_temp_var_1259 == 2 ? indices.field_2 : (_temp_var_1259 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_843_ is already defined
#ifndef _block_k_843__func
#define _block_k_843__func
__device__ int _block_k_843_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1262 = ((({ int _temp_var_1263 = ((({ int _temp_var_1264 = ((i % 4));
        (_temp_var_1264 == 0 ? indices.field_0 : (_temp_var_1264 == 1 ? indices.field_1 : (_temp_var_1264 == 2 ? indices.field_2 : (_temp_var_1264 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1263 == 0 ? indices.field_0 : (_temp_var_1263 == 1 ? indices.field_1 : (_temp_var_1263 == 2 ? indices.field_2 : (_temp_var_1263 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1262 == 0 ? indices.field_0 : (_temp_var_1262 == 1 ? indices.field_1 : (_temp_var_1262 == 2 ? indices.field_2 : (_temp_var_1262 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_845_ is already defined
#ifndef _block_k_845__func
#define _block_k_845__func
__device__ int _block_k_845_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1265 = ((({ int _temp_var_1266 = ((({ int _temp_var_1267 = ((i % 4));
        (_temp_var_1267 == 0 ? indices.field_0 : (_temp_var_1267 == 1 ? indices.field_1 : (_temp_var_1267 == 2 ? indices.field_2 : (_temp_var_1267 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1266 == 0 ? indices.field_0 : (_temp_var_1266 == 1 ? indices.field_1 : (_temp_var_1266 == 2 ? indices.field_2 : (_temp_var_1266 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1265 == 0 ? indices.field_0 : (_temp_var_1265 == 1 ? indices.field_1 : (_temp_var_1265 == 2 ? indices.field_2 : (_temp_var_1265 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_847_ is already defined
#ifndef _block_k_847__func
#define _block_k_847__func
__device__ int _block_k_847_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1268 = ((({ int _temp_var_1269 = ((({ int _temp_var_1270 = ((i % 4));
        (_temp_var_1270 == 0 ? indices.field_0 : (_temp_var_1270 == 1 ? indices.field_1 : (_temp_var_1270 == 2 ? indices.field_2 : (_temp_var_1270 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1269 == 0 ? indices.field_0 : (_temp_var_1269 == 1 ? indices.field_1 : (_temp_var_1269 == 2 ? indices.field_2 : (_temp_var_1269 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1268 == 0 ? indices.field_0 : (_temp_var_1268 == 1 ? indices.field_1 : (_temp_var_1268 == 2 ? indices.field_2 : (_temp_var_1268 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_849_ is already defined
#ifndef _block_k_849__func
#define _block_k_849__func
__device__ int _block_k_849_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1271 = ((({ int _temp_var_1272 = ((({ int _temp_var_1273 = ((i % 4));
        (_temp_var_1273 == 0 ? indices.field_0 : (_temp_var_1273 == 1 ? indices.field_1 : (_temp_var_1273 == 2 ? indices.field_2 : (_temp_var_1273 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1272 == 0 ? indices.field_0 : (_temp_var_1272 == 1 ? indices.field_1 : (_temp_var_1272 == 2 ? indices.field_2 : (_temp_var_1272 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1271 == 0 ? indices.field_0 : (_temp_var_1271 == 1 ? indices.field_1 : (_temp_var_1271 == 2 ? indices.field_2 : (_temp_var_1271 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_851_ is already defined
#ifndef _block_k_851__func
#define _block_k_851__func
__device__ int _block_k_851_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1274 = ((({ int _temp_var_1275 = ((({ int _temp_var_1276 = ((i % 4));
        (_temp_var_1276 == 0 ? indices.field_0 : (_temp_var_1276 == 1 ? indices.field_1 : (_temp_var_1276 == 2 ? indices.field_2 : (_temp_var_1276 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1275 == 0 ? indices.field_0 : (_temp_var_1275 == 1 ? indices.field_1 : (_temp_var_1275 == 2 ? indices.field_2 : (_temp_var_1275 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1274 == 0 ? indices.field_0 : (_temp_var_1274 == 1 ? indices.field_1 : (_temp_var_1274 == 2 ? indices.field_2 : (_temp_var_1274 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_853_ is already defined
#ifndef _block_k_853__func
#define _block_k_853__func
__device__ int _block_k_853_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1277 = ((({ int _temp_var_1278 = ((({ int _temp_var_1279 = ((i % 4));
        (_temp_var_1279 == 0 ? indices.field_0 : (_temp_var_1279 == 1 ? indices.field_1 : (_temp_var_1279 == 2 ? indices.field_2 : (_temp_var_1279 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1278 == 0 ? indices.field_0 : (_temp_var_1278 == 1 ? indices.field_1 : (_temp_var_1278 == 2 ? indices.field_2 : (_temp_var_1278 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1277 == 0 ? indices.field_0 : (_temp_var_1277 == 1 ? indices.field_1 : (_temp_var_1277 == 2 ? indices.field_2 : (_temp_var_1277 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_855_ is already defined
#ifndef _block_k_855__func
#define _block_k_855__func
__device__ int _block_k_855_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1280 = ((({ int _temp_var_1281 = ((({ int _temp_var_1282 = ((i % 4));
        (_temp_var_1282 == 0 ? indices.field_0 : (_temp_var_1282 == 1 ? indices.field_1 : (_temp_var_1282 == 2 ? indices.field_2 : (_temp_var_1282 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1281 == 0 ? indices.field_0 : (_temp_var_1281 == 1 ? indices.field_1 : (_temp_var_1281 == 2 ? indices.field_2 : (_temp_var_1281 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1280 == 0 ? indices.field_0 : (_temp_var_1280 == 1 ? indices.field_1 : (_temp_var_1280 == 2 ? indices.field_2 : (_temp_var_1280 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_857_ is already defined
#ifndef _block_k_857__func
#define _block_k_857__func
__device__ int _block_k_857_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1283 = ((({ int _temp_var_1284 = ((({ int _temp_var_1285 = ((i % 4));
        (_temp_var_1285 == 0 ? indices.field_0 : (_temp_var_1285 == 1 ? indices.field_1 : (_temp_var_1285 == 2 ? indices.field_2 : (_temp_var_1285 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1284 == 0 ? indices.field_0 : (_temp_var_1284 == 1 ? indices.field_1 : (_temp_var_1284 == 2 ? indices.field_2 : (_temp_var_1284 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1283 == 0 ? indices.field_0 : (_temp_var_1283 == 1 ? indices.field_1 : (_temp_var_1283 == 2 ? indices.field_2 : (_temp_var_1283 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_859_ is already defined
#ifndef _block_k_859__func
#define _block_k_859__func
__device__ int _block_k_859_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1286 = ((({ int _temp_var_1287 = ((({ int _temp_var_1288 = ((i % 4));
        (_temp_var_1288 == 0 ? indices.field_0 : (_temp_var_1288 == 1 ? indices.field_1 : (_temp_var_1288 == 2 ? indices.field_2 : (_temp_var_1288 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1287 == 0 ? indices.field_0 : (_temp_var_1287 == 1 ? indices.field_1 : (_temp_var_1287 == 2 ? indices.field_2 : (_temp_var_1287 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1286 == 0 ? indices.field_0 : (_temp_var_1286 == 1 ? indices.field_1 : (_temp_var_1286 == 2 ? indices.field_2 : (_temp_var_1286 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_861_ is already defined
#ifndef _block_k_861__func
#define _block_k_861__func
__device__ int _block_k_861_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1289 = ((({ int _temp_var_1290 = ((({ int _temp_var_1291 = ((i % 4));
        (_temp_var_1291 == 0 ? indices.field_0 : (_temp_var_1291 == 1 ? indices.field_1 : (_temp_var_1291 == 2 ? indices.field_2 : (_temp_var_1291 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1290 == 0 ? indices.field_0 : (_temp_var_1290 == 1 ? indices.field_1 : (_temp_var_1290 == 2 ? indices.field_2 : (_temp_var_1290 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1289 == 0 ? indices.field_0 : (_temp_var_1289 == 1 ? indices.field_1 : (_temp_var_1289 == 2 ? indices.field_2 : (_temp_var_1289 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_863_ is already defined
#ifndef _block_k_863__func
#define _block_k_863__func
__device__ int _block_k_863_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1292 = ((({ int _temp_var_1293 = ((({ int _temp_var_1294 = ((i % 4));
        (_temp_var_1294 == 0 ? indices.field_0 : (_temp_var_1294 == 1 ? indices.field_1 : (_temp_var_1294 == 2 ? indices.field_2 : (_temp_var_1294 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1293 == 0 ? indices.field_0 : (_temp_var_1293 == 1 ? indices.field_1 : (_temp_var_1293 == 2 ? indices.field_2 : (_temp_var_1293 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1292 == 0 ? indices.field_0 : (_temp_var_1292 == 1 ? indices.field_1 : (_temp_var_1292 == 2 ? indices.field_2 : (_temp_var_1292 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_865_ is already defined
#ifndef _block_k_865__func
#define _block_k_865__func
__device__ int _block_k_865_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1295 = ((({ int _temp_var_1296 = ((({ int _temp_var_1297 = ((i % 4));
        (_temp_var_1297 == 0 ? indices.field_0 : (_temp_var_1297 == 1 ? indices.field_1 : (_temp_var_1297 == 2 ? indices.field_2 : (_temp_var_1297 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1296 == 0 ? indices.field_0 : (_temp_var_1296 == 1 ? indices.field_1 : (_temp_var_1296 == 2 ? indices.field_2 : (_temp_var_1296 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1295 == 0 ? indices.field_0 : (_temp_var_1295 == 1 ? indices.field_1 : (_temp_var_1295 == 2 ? indices.field_2 : (_temp_var_1295 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_867_ is already defined
#ifndef _block_k_867__func
#define _block_k_867__func
__device__ int _block_k_867_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1298 = ((({ int _temp_var_1299 = ((({ int _temp_var_1300 = ((i % 4));
        (_temp_var_1300 == 0 ? indices.field_0 : (_temp_var_1300 == 1 ? indices.field_1 : (_temp_var_1300 == 2 ? indices.field_2 : (_temp_var_1300 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1299 == 0 ? indices.field_0 : (_temp_var_1299 == 1 ? indices.field_1 : (_temp_var_1299 == 2 ? indices.field_2 : (_temp_var_1299 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1298 == 0 ? indices.field_0 : (_temp_var_1298 == 1 ? indices.field_1 : (_temp_var_1298 == 2 ? indices.field_2 : (_temp_var_1298 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_869_ is already defined
#ifndef _block_k_869__func
#define _block_k_869__func
__device__ int _block_k_869_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1301 = ((({ int _temp_var_1302 = ((({ int _temp_var_1303 = ((i % 4));
        (_temp_var_1303 == 0 ? indices.field_0 : (_temp_var_1303 == 1 ? indices.field_1 : (_temp_var_1303 == 2 ? indices.field_2 : (_temp_var_1303 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1302 == 0 ? indices.field_0 : (_temp_var_1302 == 1 ? indices.field_1 : (_temp_var_1302 == 2 ? indices.field_2 : (_temp_var_1302 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1301 == 0 ? indices.field_0 : (_temp_var_1301 == 1 ? indices.field_1 : (_temp_var_1301 == 2 ? indices.field_2 : (_temp_var_1301 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_871_ is already defined
#ifndef _block_k_871__func
#define _block_k_871__func
__device__ int _block_k_871_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1304 = ((({ int _temp_var_1305 = ((({ int _temp_var_1306 = ((i % 4));
        (_temp_var_1306 == 0 ? indices.field_0 : (_temp_var_1306 == 1 ? indices.field_1 : (_temp_var_1306 == 2 ? indices.field_2 : (_temp_var_1306 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1305 == 0 ? indices.field_0 : (_temp_var_1305 == 1 ? indices.field_1 : (_temp_var_1305 == 2 ? indices.field_2 : (_temp_var_1305 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1304 == 0 ? indices.field_0 : (_temp_var_1304 == 1 ? indices.field_1 : (_temp_var_1304 == 2 ? indices.field_2 : (_temp_var_1304 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_873_ is already defined
#ifndef _block_k_873__func
#define _block_k_873__func
__device__ int _block_k_873_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1307 = ((({ int _temp_var_1308 = ((({ int _temp_var_1309 = ((i % 4));
        (_temp_var_1309 == 0 ? indices.field_0 : (_temp_var_1309 == 1 ? indices.field_1 : (_temp_var_1309 == 2 ? indices.field_2 : (_temp_var_1309 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1308 == 0 ? indices.field_0 : (_temp_var_1308 == 1 ? indices.field_1 : (_temp_var_1308 == 2 ? indices.field_2 : (_temp_var_1308 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1307 == 0 ? indices.field_0 : (_temp_var_1307 == 1 ? indices.field_1 : (_temp_var_1307 == 2 ? indices.field_2 : (_temp_var_1307 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_875_ is already defined
#ifndef _block_k_875__func
#define _block_k_875__func
__device__ int _block_k_875_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1310 = ((({ int _temp_var_1311 = ((({ int _temp_var_1312 = ((i % 4));
        (_temp_var_1312 == 0 ? indices.field_0 : (_temp_var_1312 == 1 ? indices.field_1 : (_temp_var_1312 == 2 ? indices.field_2 : (_temp_var_1312 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1311 == 0 ? indices.field_0 : (_temp_var_1311 == 1 ? indices.field_1 : (_temp_var_1311 == 2 ? indices.field_2 : (_temp_var_1311 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1310 == 0 ? indices.field_0 : (_temp_var_1310 == 1 ? indices.field_1 : (_temp_var_1310 == 2 ? indices.field_2 : (_temp_var_1310 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_877_ is already defined
#ifndef _block_k_877__func
#define _block_k_877__func
__device__ int _block_k_877_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1313 = ((({ int _temp_var_1314 = ((({ int _temp_var_1315 = ((i % 4));
        (_temp_var_1315 == 0 ? indices.field_0 : (_temp_var_1315 == 1 ? indices.field_1 : (_temp_var_1315 == 2 ? indices.field_2 : (_temp_var_1315 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1314 == 0 ? indices.field_0 : (_temp_var_1314 == 1 ? indices.field_1 : (_temp_var_1314 == 2 ? indices.field_2 : (_temp_var_1314 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1313 == 0 ? indices.field_0 : (_temp_var_1313 == 1 ? indices.field_1 : (_temp_var_1313 == 2 ? indices.field_2 : (_temp_var_1313 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_879_ is already defined
#ifndef _block_k_879__func
#define _block_k_879__func
__device__ int _block_k_879_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1316 = ((({ int _temp_var_1317 = ((({ int _temp_var_1318 = ((i % 4));
        (_temp_var_1318 == 0 ? indices.field_0 : (_temp_var_1318 == 1 ? indices.field_1 : (_temp_var_1318 == 2 ? indices.field_2 : (_temp_var_1318 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1317 == 0 ? indices.field_0 : (_temp_var_1317 == 1 ? indices.field_1 : (_temp_var_1317 == 2 ? indices.field_2 : (_temp_var_1317 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1316 == 0 ? indices.field_0 : (_temp_var_1316 == 1 ? indices.field_1 : (_temp_var_1316 == 2 ? indices.field_2 : (_temp_var_1316 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_881_ is already defined
#ifndef _block_k_881__func
#define _block_k_881__func
__device__ int _block_k_881_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1319 = ((({ int _temp_var_1320 = ((({ int _temp_var_1321 = ((i % 4));
        (_temp_var_1321 == 0 ? indices.field_0 : (_temp_var_1321 == 1 ? indices.field_1 : (_temp_var_1321 == 2 ? indices.field_2 : (_temp_var_1321 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1320 == 0 ? indices.field_0 : (_temp_var_1320 == 1 ? indices.field_1 : (_temp_var_1320 == 2 ? indices.field_2 : (_temp_var_1320 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1319 == 0 ? indices.field_0 : (_temp_var_1319 == 1 ? indices.field_1 : (_temp_var_1319 == 2 ? indices.field_2 : (_temp_var_1319 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_883_ is already defined
#ifndef _block_k_883__func
#define _block_k_883__func
__device__ int _block_k_883_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1322 = ((({ int _temp_var_1323 = ((({ int _temp_var_1324 = ((i % 4));
        (_temp_var_1324 == 0 ? indices.field_0 : (_temp_var_1324 == 1 ? indices.field_1 : (_temp_var_1324 == 2 ? indices.field_2 : (_temp_var_1324 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1323 == 0 ? indices.field_0 : (_temp_var_1323 == 1 ? indices.field_1 : (_temp_var_1323 == 2 ? indices.field_2 : (_temp_var_1323 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1322 == 0 ? indices.field_0 : (_temp_var_1322 == 1 ? indices.field_1 : (_temp_var_1322 == 2 ? indices.field_2 : (_temp_var_1322 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_885_ is already defined
#ifndef _block_k_885__func
#define _block_k_885__func
__device__ int _block_k_885_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1325 = ((({ int _temp_var_1326 = ((({ int _temp_var_1327 = ((i % 4));
        (_temp_var_1327 == 0 ? indices.field_0 : (_temp_var_1327 == 1 ? indices.field_1 : (_temp_var_1327 == 2 ? indices.field_2 : (_temp_var_1327 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1326 == 0 ? indices.field_0 : (_temp_var_1326 == 1 ? indices.field_1 : (_temp_var_1326 == 2 ? indices.field_2 : (_temp_var_1326 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1325 == 0 ? indices.field_0 : (_temp_var_1325 == 1 ? indices.field_1 : (_temp_var_1325 == 2 ? indices.field_2 : (_temp_var_1325 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_887_ is already defined
#ifndef _block_k_887__func
#define _block_k_887__func
__device__ int _block_k_887_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1328 = ((({ int _temp_var_1329 = ((({ int _temp_var_1330 = ((i % 4));
        (_temp_var_1330 == 0 ? indices.field_0 : (_temp_var_1330 == 1 ? indices.field_1 : (_temp_var_1330 == 2 ? indices.field_2 : (_temp_var_1330 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1329 == 0 ? indices.field_0 : (_temp_var_1329 == 1 ? indices.field_1 : (_temp_var_1329 == 2 ? indices.field_2 : (_temp_var_1329 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1328 == 0 ? indices.field_0 : (_temp_var_1328 == 1 ? indices.field_1 : (_temp_var_1328 == 2 ? indices.field_2 : (_temp_var_1328 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_889_ is already defined
#ifndef _block_k_889__func
#define _block_k_889__func
__device__ int _block_k_889_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1331 = ((({ int _temp_var_1332 = ((({ int _temp_var_1333 = ((i % 4));
        (_temp_var_1333 == 0 ? indices.field_0 : (_temp_var_1333 == 1 ? indices.field_1 : (_temp_var_1333 == 2 ? indices.field_2 : (_temp_var_1333 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1332 == 0 ? indices.field_0 : (_temp_var_1332 == 1 ? indices.field_1 : (_temp_var_1332 == 2 ? indices.field_2 : (_temp_var_1332 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1331 == 0 ? indices.field_0 : (_temp_var_1331 == 1 ? indices.field_1 : (_temp_var_1331 == 2 ? indices.field_2 : (_temp_var_1331 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_891_ is already defined
#ifndef _block_k_891__func
#define _block_k_891__func
__device__ int _block_k_891_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1334 = ((({ int _temp_var_1335 = ((({ int _temp_var_1336 = ((i % 4));
        (_temp_var_1336 == 0 ? indices.field_0 : (_temp_var_1336 == 1 ? indices.field_1 : (_temp_var_1336 == 2 ? indices.field_2 : (_temp_var_1336 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1335 == 0 ? indices.field_0 : (_temp_var_1335 == 1 ? indices.field_1 : (_temp_var_1335 == 2 ? indices.field_2 : (_temp_var_1335 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1334 == 0 ? indices.field_0 : (_temp_var_1334 == 1 ? indices.field_1 : (_temp_var_1334 == 2 ? indices.field_2 : (_temp_var_1334 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_893_ is already defined
#ifndef _block_k_893__func
#define _block_k_893__func
__device__ int _block_k_893_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1337 = ((({ int _temp_var_1338 = ((({ int _temp_var_1339 = ((i % 4));
        (_temp_var_1339 == 0 ? indices.field_0 : (_temp_var_1339 == 1 ? indices.field_1 : (_temp_var_1339 == 2 ? indices.field_2 : (_temp_var_1339 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1338 == 0 ? indices.field_0 : (_temp_var_1338 == 1 ? indices.field_1 : (_temp_var_1338 == 2 ? indices.field_2 : (_temp_var_1338 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1337 == 0 ? indices.field_0 : (_temp_var_1337 == 1 ? indices.field_1 : (_temp_var_1337 == 2 ? indices.field_2 : (_temp_var_1337 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_895_ is already defined
#ifndef _block_k_895__func
#define _block_k_895__func
__device__ int _block_k_895_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1340 = ((({ int _temp_var_1341 = ((({ int _temp_var_1342 = ((i % 4));
        (_temp_var_1342 == 0 ? indices.field_0 : (_temp_var_1342 == 1 ? indices.field_1 : (_temp_var_1342 == 2 ? indices.field_2 : (_temp_var_1342 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1341 == 0 ? indices.field_0 : (_temp_var_1341 == 1 ? indices.field_1 : (_temp_var_1341 == 2 ? indices.field_2 : (_temp_var_1341 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1340 == 0 ? indices.field_0 : (_temp_var_1340 == 1 ? indices.field_1 : (_temp_var_1340 == 2 ? indices.field_2 : (_temp_var_1340 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_897_ is already defined
#ifndef _block_k_897__func
#define _block_k_897__func
__device__ int _block_k_897_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1343 = ((({ int _temp_var_1344 = ((({ int _temp_var_1345 = ((i % 4));
        (_temp_var_1345 == 0 ? indices.field_0 : (_temp_var_1345 == 1 ? indices.field_1 : (_temp_var_1345 == 2 ? indices.field_2 : (_temp_var_1345 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1344 == 0 ? indices.field_0 : (_temp_var_1344 == 1 ? indices.field_1 : (_temp_var_1344 == 2 ? indices.field_2 : (_temp_var_1344 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1343 == 0 ? indices.field_0 : (_temp_var_1343 == 1 ? indices.field_1 : (_temp_var_1343 == 2 ? indices.field_2 : (_temp_var_1343 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_899_ is already defined
#ifndef _block_k_899__func
#define _block_k_899__func
__device__ int _block_k_899_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1346 = ((({ int _temp_var_1347 = ((({ int _temp_var_1348 = ((i % 4));
        (_temp_var_1348 == 0 ? indices.field_0 : (_temp_var_1348 == 1 ? indices.field_1 : (_temp_var_1348 == 2 ? indices.field_2 : (_temp_var_1348 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1347 == 0 ? indices.field_0 : (_temp_var_1347 == 1 ? indices.field_1 : (_temp_var_1347 == 2 ? indices.field_2 : (_temp_var_1347 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1346 == 0 ? indices.field_0 : (_temp_var_1346 == 1 ? indices.field_1 : (_temp_var_1346 == 2 ? indices.field_2 : (_temp_var_1346 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_901_ is already defined
#ifndef _block_k_901__func
#define _block_k_901__func
__device__ int _block_k_901_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1349 = ((({ int _temp_var_1350 = ((({ int _temp_var_1351 = ((i % 4));
        (_temp_var_1351 == 0 ? indices.field_0 : (_temp_var_1351 == 1 ? indices.field_1 : (_temp_var_1351 == 2 ? indices.field_2 : (_temp_var_1351 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1350 == 0 ? indices.field_0 : (_temp_var_1350 == 1 ? indices.field_1 : (_temp_var_1350 == 2 ? indices.field_2 : (_temp_var_1350 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1349 == 0 ? indices.field_0 : (_temp_var_1349 == 1 ? indices.field_1 : (_temp_var_1349 == 2 ? indices.field_2 : (_temp_var_1349 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_903_ is already defined
#ifndef _block_k_903__func
#define _block_k_903__func
__device__ int _block_k_903_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1352 = ((({ int _temp_var_1353 = ((({ int _temp_var_1354 = ((i % 4));
        (_temp_var_1354 == 0 ? indices.field_0 : (_temp_var_1354 == 1 ? indices.field_1 : (_temp_var_1354 == 2 ? indices.field_2 : (_temp_var_1354 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1353 == 0 ? indices.field_0 : (_temp_var_1353 == 1 ? indices.field_1 : (_temp_var_1353 == 2 ? indices.field_2 : (_temp_var_1353 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1352 == 0 ? indices.field_0 : (_temp_var_1352 == 1 ? indices.field_1 : (_temp_var_1352 == 2 ? indices.field_2 : (_temp_var_1352 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_905_ is already defined
#ifndef _block_k_905__func
#define _block_k_905__func
__device__ int _block_k_905_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1355 = ((({ int _temp_var_1356 = ((({ int _temp_var_1357 = ((i % 4));
        (_temp_var_1357 == 0 ? indices.field_0 : (_temp_var_1357 == 1 ? indices.field_1 : (_temp_var_1357 == 2 ? indices.field_2 : (_temp_var_1357 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1356 == 0 ? indices.field_0 : (_temp_var_1356 == 1 ? indices.field_1 : (_temp_var_1356 == 2 ? indices.field_2 : (_temp_var_1356 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1355 == 0 ? indices.field_0 : (_temp_var_1355 == 1 ? indices.field_1 : (_temp_var_1355 == 2 ? indices.field_2 : (_temp_var_1355 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_907_ is already defined
#ifndef _block_k_907__func
#define _block_k_907__func
__device__ int _block_k_907_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1358 = ((({ int _temp_var_1359 = ((({ int _temp_var_1360 = ((i % 4));
        (_temp_var_1360 == 0 ? indices.field_0 : (_temp_var_1360 == 1 ? indices.field_1 : (_temp_var_1360 == 2 ? indices.field_2 : (_temp_var_1360 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1359 == 0 ? indices.field_0 : (_temp_var_1359 == 1 ? indices.field_1 : (_temp_var_1359 == 2 ? indices.field_2 : (_temp_var_1359 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1358 == 0 ? indices.field_0 : (_temp_var_1358 == 1 ? indices.field_1 : (_temp_var_1358 == 2 ? indices.field_2 : (_temp_var_1358 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_909_ is already defined
#ifndef _block_k_909__func
#define _block_k_909__func
__device__ int _block_k_909_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1361 = ((({ int _temp_var_1362 = ((({ int _temp_var_1363 = ((i % 4));
        (_temp_var_1363 == 0 ? indices.field_0 : (_temp_var_1363 == 1 ? indices.field_1 : (_temp_var_1363 == 2 ? indices.field_2 : (_temp_var_1363 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1362 == 0 ? indices.field_0 : (_temp_var_1362 == 1 ? indices.field_1 : (_temp_var_1362 == 2 ? indices.field_2 : (_temp_var_1362 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1361 == 0 ? indices.field_0 : (_temp_var_1361 == 1 ? indices.field_1 : (_temp_var_1361 == 2 ? indices.field_2 : (_temp_var_1361 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_911_ is already defined
#ifndef _block_k_911__func
#define _block_k_911__func
__device__ int _block_k_911_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1364 = ((({ int _temp_var_1365 = ((({ int _temp_var_1366 = ((i % 4));
        (_temp_var_1366 == 0 ? indices.field_0 : (_temp_var_1366 == 1 ? indices.field_1 : (_temp_var_1366 == 2 ? indices.field_2 : (_temp_var_1366 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1365 == 0 ? indices.field_0 : (_temp_var_1365 == 1 ? indices.field_1 : (_temp_var_1365 == 2 ? indices.field_2 : (_temp_var_1365 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1364 == 0 ? indices.field_0 : (_temp_var_1364 == 1 ? indices.field_1 : (_temp_var_1364 == 2 ? indices.field_2 : (_temp_var_1364 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_913_ is already defined
#ifndef _block_k_913__func
#define _block_k_913__func
__device__ int _block_k_913_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1367 = ((({ int _temp_var_1368 = ((({ int _temp_var_1369 = ((i % 4));
        (_temp_var_1369 == 0 ? indices.field_0 : (_temp_var_1369 == 1 ? indices.field_1 : (_temp_var_1369 == 2 ? indices.field_2 : (_temp_var_1369 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1368 == 0 ? indices.field_0 : (_temp_var_1368 == 1 ? indices.field_1 : (_temp_var_1368 == 2 ? indices.field_2 : (_temp_var_1368 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1367 == 0 ? indices.field_0 : (_temp_var_1367 == 1 ? indices.field_1 : (_temp_var_1367 == 2 ? indices.field_2 : (_temp_var_1367 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_915_ is already defined
#ifndef _block_k_915__func
#define _block_k_915__func
__device__ int _block_k_915_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1370 = ((({ int _temp_var_1371 = ((({ int _temp_var_1372 = ((i % 4));
        (_temp_var_1372 == 0 ? indices.field_0 : (_temp_var_1372 == 1 ? indices.field_1 : (_temp_var_1372 == 2 ? indices.field_2 : (_temp_var_1372 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1371 == 0 ? indices.field_0 : (_temp_var_1371 == 1 ? indices.field_1 : (_temp_var_1371 == 2 ? indices.field_2 : (_temp_var_1371 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1370 == 0 ? indices.field_0 : (_temp_var_1370 == 1 ? indices.field_1 : (_temp_var_1370 == 2 ? indices.field_2 : (_temp_var_1370 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_917_ is already defined
#ifndef _block_k_917__func
#define _block_k_917__func
__device__ int _block_k_917_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1373 = ((({ int _temp_var_1374 = ((({ int _temp_var_1375 = ((i % 4));
        (_temp_var_1375 == 0 ? indices.field_0 : (_temp_var_1375 == 1 ? indices.field_1 : (_temp_var_1375 == 2 ? indices.field_2 : (_temp_var_1375 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1374 == 0 ? indices.field_0 : (_temp_var_1374 == 1 ? indices.field_1 : (_temp_var_1374 == 2 ? indices.field_2 : (_temp_var_1374 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1373 == 0 ? indices.field_0 : (_temp_var_1373 == 1 ? indices.field_1 : (_temp_var_1373 == 2 ? indices.field_2 : (_temp_var_1373 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_919_ is already defined
#ifndef _block_k_919__func
#define _block_k_919__func
__device__ int _block_k_919_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1376 = ((({ int _temp_var_1377 = ((({ int _temp_var_1378 = ((i % 4));
        (_temp_var_1378 == 0 ? indices.field_0 : (_temp_var_1378 == 1 ? indices.field_1 : (_temp_var_1378 == 2 ? indices.field_2 : (_temp_var_1378 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1377 == 0 ? indices.field_0 : (_temp_var_1377 == 1 ? indices.field_1 : (_temp_var_1377 == 2 ? indices.field_2 : (_temp_var_1377 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1376 == 0 ? indices.field_0 : (_temp_var_1376 == 1 ? indices.field_1 : (_temp_var_1376 == 2 ? indices.field_2 : (_temp_var_1376 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_921_ is already defined
#ifndef _block_k_921__func
#define _block_k_921__func
__device__ int _block_k_921_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1379 = ((({ int _temp_var_1380 = ((({ int _temp_var_1381 = ((i % 4));
        (_temp_var_1381 == 0 ? indices.field_0 : (_temp_var_1381 == 1 ? indices.field_1 : (_temp_var_1381 == 2 ? indices.field_2 : (_temp_var_1381 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1380 == 0 ? indices.field_0 : (_temp_var_1380 == 1 ? indices.field_1 : (_temp_var_1380 == 2 ? indices.field_2 : (_temp_var_1380 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1379 == 0 ? indices.field_0 : (_temp_var_1379 == 1 ? indices.field_1 : (_temp_var_1379 == 2 ? indices.field_2 : (_temp_var_1379 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_923_ is already defined
#ifndef _block_k_923__func
#define _block_k_923__func
__device__ int _block_k_923_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1382 = ((({ int _temp_var_1383 = ((({ int _temp_var_1384 = ((i % 4));
        (_temp_var_1384 == 0 ? indices.field_0 : (_temp_var_1384 == 1 ? indices.field_1 : (_temp_var_1384 == 2 ? indices.field_2 : (_temp_var_1384 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1383 == 0 ? indices.field_0 : (_temp_var_1383 == 1 ? indices.field_1 : (_temp_var_1383 == 2 ? indices.field_2 : (_temp_var_1383 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1382 == 0 ? indices.field_0 : (_temp_var_1382 == 1 ? indices.field_1 : (_temp_var_1382 == 2 ? indices.field_2 : (_temp_var_1382 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_925_ is already defined
#ifndef _block_k_925__func
#define _block_k_925__func
__device__ int _block_k_925_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1385 = ((({ int _temp_var_1386 = ((({ int _temp_var_1387 = ((i % 4));
        (_temp_var_1387 == 0 ? indices.field_0 : (_temp_var_1387 == 1 ? indices.field_1 : (_temp_var_1387 == 2 ? indices.field_2 : (_temp_var_1387 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1386 == 0 ? indices.field_0 : (_temp_var_1386 == 1 ? indices.field_1 : (_temp_var_1386 == 2 ? indices.field_2 : (_temp_var_1386 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1385 == 0 ? indices.field_0 : (_temp_var_1385 == 1 ? indices.field_1 : (_temp_var_1385 == 2 ? indices.field_2 : (_temp_var_1385 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_927_ is already defined
#ifndef _block_k_927__func
#define _block_k_927__func
__device__ int _block_k_927_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1388 = ((({ int _temp_var_1389 = ((({ int _temp_var_1390 = ((i % 4));
        (_temp_var_1390 == 0 ? indices.field_0 : (_temp_var_1390 == 1 ? indices.field_1 : (_temp_var_1390 == 2 ? indices.field_2 : (_temp_var_1390 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1389 == 0 ? indices.field_0 : (_temp_var_1389 == 1 ? indices.field_1 : (_temp_var_1389 == 2 ? indices.field_2 : (_temp_var_1389 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1388 == 0 ? indices.field_0 : (_temp_var_1388 == 1 ? indices.field_1 : (_temp_var_1388 == 2 ? indices.field_2 : (_temp_var_1388 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_929_ is already defined
#ifndef _block_k_929__func
#define _block_k_929__func
__device__ int _block_k_929_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1391 = ((({ int _temp_var_1392 = ((({ int _temp_var_1393 = ((i % 4));
        (_temp_var_1393 == 0 ? indices.field_0 : (_temp_var_1393 == 1 ? indices.field_1 : (_temp_var_1393 == 2 ? indices.field_2 : (_temp_var_1393 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1392 == 0 ? indices.field_0 : (_temp_var_1392 == 1 ? indices.field_1 : (_temp_var_1392 == 2 ? indices.field_2 : (_temp_var_1392 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1391 == 0 ? indices.field_0 : (_temp_var_1391 == 1 ? indices.field_1 : (_temp_var_1391 == 2 ? indices.field_2 : (_temp_var_1391 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_931_ is already defined
#ifndef _block_k_931__func
#define _block_k_931__func
__device__ int _block_k_931_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1394 = ((({ int _temp_var_1395 = ((({ int _temp_var_1396 = ((i % 4));
        (_temp_var_1396 == 0 ? indices.field_0 : (_temp_var_1396 == 1 ? indices.field_1 : (_temp_var_1396 == 2 ? indices.field_2 : (_temp_var_1396 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1395 == 0 ? indices.field_0 : (_temp_var_1395 == 1 ? indices.field_1 : (_temp_var_1395 == 2 ? indices.field_2 : (_temp_var_1395 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1394 == 0 ? indices.field_0 : (_temp_var_1394 == 1 ? indices.field_1 : (_temp_var_1394 == 2 ? indices.field_2 : (_temp_var_1394 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_933_ is already defined
#ifndef _block_k_933__func
#define _block_k_933__func
__device__ int _block_k_933_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1397 = ((({ int _temp_var_1398 = ((({ int _temp_var_1399 = ((i % 4));
        (_temp_var_1399 == 0 ? indices.field_0 : (_temp_var_1399 == 1 ? indices.field_1 : (_temp_var_1399 == 2 ? indices.field_2 : (_temp_var_1399 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1398 == 0 ? indices.field_0 : (_temp_var_1398 == 1 ? indices.field_1 : (_temp_var_1398 == 2 ? indices.field_2 : (_temp_var_1398 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1397 == 0 ? indices.field_0 : (_temp_var_1397 == 1 ? indices.field_1 : (_temp_var_1397 == 2 ? indices.field_2 : (_temp_var_1397 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_935_ is already defined
#ifndef _block_k_935__func
#define _block_k_935__func
__device__ int _block_k_935_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1400 = ((({ int _temp_var_1401 = ((({ int _temp_var_1402 = ((i % 4));
        (_temp_var_1402 == 0 ? indices.field_0 : (_temp_var_1402 == 1 ? indices.field_1 : (_temp_var_1402 == 2 ? indices.field_2 : (_temp_var_1402 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1401 == 0 ? indices.field_0 : (_temp_var_1401 == 1 ? indices.field_1 : (_temp_var_1401 == 2 ? indices.field_2 : (_temp_var_1401 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1400 == 0 ? indices.field_0 : (_temp_var_1400 == 1 ? indices.field_1 : (_temp_var_1400 == 2 ? indices.field_2 : (_temp_var_1400 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_937_ is already defined
#ifndef _block_k_937__func
#define _block_k_937__func
__device__ int _block_k_937_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1403 = ((({ int _temp_var_1404 = ((({ int _temp_var_1405 = ((i % 4));
        (_temp_var_1405 == 0 ? indices.field_0 : (_temp_var_1405 == 1 ? indices.field_1 : (_temp_var_1405 == 2 ? indices.field_2 : (_temp_var_1405 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1404 == 0 ? indices.field_0 : (_temp_var_1404 == 1 ? indices.field_1 : (_temp_var_1404 == 2 ? indices.field_2 : (_temp_var_1404 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1403 == 0 ? indices.field_0 : (_temp_var_1403 == 1 ? indices.field_1 : (_temp_var_1403 == 2 ? indices.field_2 : (_temp_var_1403 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_939_ is already defined
#ifndef _block_k_939__func
#define _block_k_939__func
__device__ int _block_k_939_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1406 = ((({ int _temp_var_1407 = ((({ int _temp_var_1408 = ((i % 4));
        (_temp_var_1408 == 0 ? indices.field_0 : (_temp_var_1408 == 1 ? indices.field_1 : (_temp_var_1408 == 2 ? indices.field_2 : (_temp_var_1408 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1407 == 0 ? indices.field_0 : (_temp_var_1407 == 1 ? indices.field_1 : (_temp_var_1407 == 2 ? indices.field_2 : (_temp_var_1407 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1406 == 0 ? indices.field_0 : (_temp_var_1406 == 1 ? indices.field_1 : (_temp_var_1406 == 2 ? indices.field_2 : (_temp_var_1406 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_941_ is already defined
#ifndef _block_k_941__func
#define _block_k_941__func
__device__ int _block_k_941_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1409 = ((({ int _temp_var_1410 = ((({ int _temp_var_1411 = ((i % 4));
        (_temp_var_1411 == 0 ? indices.field_0 : (_temp_var_1411 == 1 ? indices.field_1 : (_temp_var_1411 == 2 ? indices.field_2 : (_temp_var_1411 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1410 == 0 ? indices.field_0 : (_temp_var_1410 == 1 ? indices.field_1 : (_temp_var_1410 == 2 ? indices.field_2 : (_temp_var_1410 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1409 == 0 ? indices.field_0 : (_temp_var_1409 == 1 ? indices.field_1 : (_temp_var_1409 == 2 ? indices.field_2 : (_temp_var_1409 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_943_ is already defined
#ifndef _block_k_943__func
#define _block_k_943__func
__device__ int _block_k_943_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1412 = ((({ int _temp_var_1413 = ((({ int _temp_var_1414 = ((i % 4));
        (_temp_var_1414 == 0 ? indices.field_0 : (_temp_var_1414 == 1 ? indices.field_1 : (_temp_var_1414 == 2 ? indices.field_2 : (_temp_var_1414 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1413 == 0 ? indices.field_0 : (_temp_var_1413 == 1 ? indices.field_1 : (_temp_var_1413 == 2 ? indices.field_2 : (_temp_var_1413 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1412 == 0 ? indices.field_0 : (_temp_var_1412 == 1 ? indices.field_1 : (_temp_var_1412 == 2 ? indices.field_2 : (_temp_var_1412 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_945_ is already defined
#ifndef _block_k_945__func
#define _block_k_945__func
__device__ int _block_k_945_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1415 = ((({ int _temp_var_1416 = ((({ int _temp_var_1417 = ((i % 4));
        (_temp_var_1417 == 0 ? indices.field_0 : (_temp_var_1417 == 1 ? indices.field_1 : (_temp_var_1417 == 2 ? indices.field_2 : (_temp_var_1417 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1416 == 0 ? indices.field_0 : (_temp_var_1416 == 1 ? indices.field_1 : (_temp_var_1416 == 2 ? indices.field_2 : (_temp_var_1416 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1415 == 0 ? indices.field_0 : (_temp_var_1415 == 1 ? indices.field_1 : (_temp_var_1415 == 2 ? indices.field_2 : (_temp_var_1415 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_947_ is already defined
#ifndef _block_k_947__func
#define _block_k_947__func
__device__ int _block_k_947_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1418 = ((({ int _temp_var_1419 = ((({ int _temp_var_1420 = ((i % 4));
        (_temp_var_1420 == 0 ? indices.field_0 : (_temp_var_1420 == 1 ? indices.field_1 : (_temp_var_1420 == 2 ? indices.field_2 : (_temp_var_1420 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1419 == 0 ? indices.field_0 : (_temp_var_1419 == 1 ? indices.field_1 : (_temp_var_1419 == 2 ? indices.field_2 : (_temp_var_1419 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1418 == 0 ? indices.field_0 : (_temp_var_1418 == 1 ? indices.field_1 : (_temp_var_1418 == 2 ? indices.field_2 : (_temp_var_1418 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_949_ is already defined
#ifndef _block_k_949__func
#define _block_k_949__func
__device__ int _block_k_949_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1421 = ((({ int _temp_var_1422 = ((({ int _temp_var_1423 = ((i % 4));
        (_temp_var_1423 == 0 ? indices.field_0 : (_temp_var_1423 == 1 ? indices.field_1 : (_temp_var_1423 == 2 ? indices.field_2 : (_temp_var_1423 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1422 == 0 ? indices.field_0 : (_temp_var_1422 == 1 ? indices.field_1 : (_temp_var_1422 == 2 ? indices.field_2 : (_temp_var_1422 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1421 == 0 ? indices.field_0 : (_temp_var_1421 == 1 ? indices.field_1 : (_temp_var_1421 == 2 ? indices.field_2 : (_temp_var_1421 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_951_ is already defined
#ifndef _block_k_951__func
#define _block_k_951__func
__device__ int _block_k_951_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1424 = ((({ int _temp_var_1425 = ((({ int _temp_var_1426 = ((i % 4));
        (_temp_var_1426 == 0 ? indices.field_0 : (_temp_var_1426 == 1 ? indices.field_1 : (_temp_var_1426 == 2 ? indices.field_2 : (_temp_var_1426 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1425 == 0 ? indices.field_0 : (_temp_var_1425 == 1 ? indices.field_1 : (_temp_var_1425 == 2 ? indices.field_2 : (_temp_var_1425 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1424 == 0 ? indices.field_0 : (_temp_var_1424 == 1 ? indices.field_1 : (_temp_var_1424 == 2 ? indices.field_2 : (_temp_var_1424 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_953_ is already defined
#ifndef _block_k_953__func
#define _block_k_953__func
__device__ int _block_k_953_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1427 = ((({ int _temp_var_1428 = ((({ int _temp_var_1429 = ((i % 4));
        (_temp_var_1429 == 0 ? indices.field_0 : (_temp_var_1429 == 1 ? indices.field_1 : (_temp_var_1429 == 2 ? indices.field_2 : (_temp_var_1429 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1428 == 0 ? indices.field_0 : (_temp_var_1428 == 1 ? indices.field_1 : (_temp_var_1428 == 2 ? indices.field_2 : (_temp_var_1428 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1427 == 0 ? indices.field_0 : (_temp_var_1427 == 1 ? indices.field_1 : (_temp_var_1427 == 2 ? indices.field_2 : (_temp_var_1427 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_955_ is already defined
#ifndef _block_k_955__func
#define _block_k_955__func
__device__ int _block_k_955_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1430 = ((({ int _temp_var_1431 = ((({ int _temp_var_1432 = ((i % 4));
        (_temp_var_1432 == 0 ? indices.field_0 : (_temp_var_1432 == 1 ? indices.field_1 : (_temp_var_1432 == 2 ? indices.field_2 : (_temp_var_1432 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1431 == 0 ? indices.field_0 : (_temp_var_1431 == 1 ? indices.field_1 : (_temp_var_1431 == 2 ? indices.field_2 : (_temp_var_1431 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1430 == 0 ? indices.field_0 : (_temp_var_1430 == 1 ? indices.field_1 : (_temp_var_1430 == 2 ? indices.field_2 : (_temp_var_1430 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_957_ is already defined
#ifndef _block_k_957__func
#define _block_k_957__func
__device__ int _block_k_957_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1433 = ((({ int _temp_var_1434 = ((({ int _temp_var_1435 = ((i % 4));
        (_temp_var_1435 == 0 ? indices.field_0 : (_temp_var_1435 == 1 ? indices.field_1 : (_temp_var_1435 == 2 ? indices.field_2 : (_temp_var_1435 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1434 == 0 ? indices.field_0 : (_temp_var_1434 == 1 ? indices.field_1 : (_temp_var_1434 == 2 ? indices.field_2 : (_temp_var_1434 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1433 == 0 ? indices.field_0 : (_temp_var_1433 == 1 ? indices.field_1 : (_temp_var_1433 == 2 ? indices.field_2 : (_temp_var_1433 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_959_ is already defined
#ifndef _block_k_959__func
#define _block_k_959__func
__device__ int _block_k_959_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1436 = ((({ int _temp_var_1437 = ((({ int _temp_var_1438 = ((i % 4));
        (_temp_var_1438 == 0 ? indices.field_0 : (_temp_var_1438 == 1 ? indices.field_1 : (_temp_var_1438 == 2 ? indices.field_2 : (_temp_var_1438 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1437 == 0 ? indices.field_0 : (_temp_var_1437 == 1 ? indices.field_1 : (_temp_var_1437 == 2 ? indices.field_2 : (_temp_var_1437 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1436 == 0 ? indices.field_0 : (_temp_var_1436 == 1 ? indices.field_1 : (_temp_var_1436 == 2 ? indices.field_2 : (_temp_var_1436 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_961_ is already defined
#ifndef _block_k_961__func
#define _block_k_961__func
__device__ int _block_k_961_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1439 = ((({ int _temp_var_1440 = ((({ int _temp_var_1441 = ((i % 4));
        (_temp_var_1441 == 0 ? indices.field_0 : (_temp_var_1441 == 1 ? indices.field_1 : (_temp_var_1441 == 2 ? indices.field_2 : (_temp_var_1441 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1440 == 0 ? indices.field_0 : (_temp_var_1440 == 1 ? indices.field_1 : (_temp_var_1440 == 2 ? indices.field_2 : (_temp_var_1440 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1439 == 0 ? indices.field_0 : (_temp_var_1439 == 1 ? indices.field_1 : (_temp_var_1439 == 2 ? indices.field_2 : (_temp_var_1439 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_963_ is already defined
#ifndef _block_k_963__func
#define _block_k_963__func
__device__ int _block_k_963_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1442 = ((({ int _temp_var_1443 = ((({ int _temp_var_1444 = ((i % 4));
        (_temp_var_1444 == 0 ? indices.field_0 : (_temp_var_1444 == 1 ? indices.field_1 : (_temp_var_1444 == 2 ? indices.field_2 : (_temp_var_1444 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1443 == 0 ? indices.field_0 : (_temp_var_1443 == 1 ? indices.field_1 : (_temp_var_1443 == 2 ? indices.field_2 : (_temp_var_1443 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1442 == 0 ? indices.field_0 : (_temp_var_1442 == 1 ? indices.field_1 : (_temp_var_1442 == 2 ? indices.field_2 : (_temp_var_1442 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_965_ is already defined
#ifndef _block_k_965__func
#define _block_k_965__func
__device__ int _block_k_965_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1445 = ((({ int _temp_var_1446 = ((({ int _temp_var_1447 = ((i % 4));
        (_temp_var_1447 == 0 ? indices.field_0 : (_temp_var_1447 == 1 ? indices.field_1 : (_temp_var_1447 == 2 ? indices.field_2 : (_temp_var_1447 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1446 == 0 ? indices.field_0 : (_temp_var_1446 == 1 ? indices.field_1 : (_temp_var_1446 == 2 ? indices.field_2 : (_temp_var_1446 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1445 == 0 ? indices.field_0 : (_temp_var_1445 == 1 ? indices.field_1 : (_temp_var_1445 == 2 ? indices.field_2 : (_temp_var_1445 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_967_ is already defined
#ifndef _block_k_967__func
#define _block_k_967__func
__device__ int _block_k_967_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1448 = ((({ int _temp_var_1449 = ((({ int _temp_var_1450 = ((i % 4));
        (_temp_var_1450 == 0 ? indices.field_0 : (_temp_var_1450 == 1 ? indices.field_1 : (_temp_var_1450 == 2 ? indices.field_2 : (_temp_var_1450 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1449 == 0 ? indices.field_0 : (_temp_var_1449 == 1 ? indices.field_1 : (_temp_var_1449 == 2 ? indices.field_2 : (_temp_var_1449 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1448 == 0 ? indices.field_0 : (_temp_var_1448 == 1 ? indices.field_1 : (_temp_var_1448 == 2 ? indices.field_2 : (_temp_var_1448 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_969_ is already defined
#ifndef _block_k_969__func
#define _block_k_969__func
__device__ int _block_k_969_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1451 = ((({ int _temp_var_1452 = ((({ int _temp_var_1453 = ((i % 4));
        (_temp_var_1453 == 0 ? indices.field_0 : (_temp_var_1453 == 1 ? indices.field_1 : (_temp_var_1453 == 2 ? indices.field_2 : (_temp_var_1453 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1452 == 0 ? indices.field_0 : (_temp_var_1452 == 1 ? indices.field_1 : (_temp_var_1452 == 2 ? indices.field_2 : (_temp_var_1452 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1451 == 0 ? indices.field_0 : (_temp_var_1451 == 1 ? indices.field_1 : (_temp_var_1451 == 2 ? indices.field_2 : (_temp_var_1451 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_971_ is already defined
#ifndef _block_k_971__func
#define _block_k_971__func
__device__ int _block_k_971_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1454 = ((({ int _temp_var_1455 = ((({ int _temp_var_1456 = ((i % 4));
        (_temp_var_1456 == 0 ? indices.field_0 : (_temp_var_1456 == 1 ? indices.field_1 : (_temp_var_1456 == 2 ? indices.field_2 : (_temp_var_1456 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1455 == 0 ? indices.field_0 : (_temp_var_1455 == 1 ? indices.field_1 : (_temp_var_1455 == 2 ? indices.field_2 : (_temp_var_1455 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1454 == 0 ? indices.field_0 : (_temp_var_1454 == 1 ? indices.field_1 : (_temp_var_1454 == 2 ? indices.field_2 : (_temp_var_1454 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_973_ is already defined
#ifndef _block_k_973__func
#define _block_k_973__func
__device__ int _block_k_973_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1457 = ((({ int _temp_var_1458 = ((({ int _temp_var_1459 = ((i % 4));
        (_temp_var_1459 == 0 ? indices.field_0 : (_temp_var_1459 == 1 ? indices.field_1 : (_temp_var_1459 == 2 ? indices.field_2 : (_temp_var_1459 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1458 == 0 ? indices.field_0 : (_temp_var_1458 == 1 ? indices.field_1 : (_temp_var_1458 == 2 ? indices.field_2 : (_temp_var_1458 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1457 == 0 ? indices.field_0 : (_temp_var_1457 == 1 ? indices.field_1 : (_temp_var_1457 == 2 ? indices.field_2 : (_temp_var_1457 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_975_ is already defined
#ifndef _block_k_975__func
#define _block_k_975__func
__device__ int _block_k_975_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1460 = ((({ int _temp_var_1461 = ((({ int _temp_var_1462 = ((i % 4));
        (_temp_var_1462 == 0 ? indices.field_0 : (_temp_var_1462 == 1 ? indices.field_1 : (_temp_var_1462 == 2 ? indices.field_2 : (_temp_var_1462 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1461 == 0 ? indices.field_0 : (_temp_var_1461 == 1 ? indices.field_1 : (_temp_var_1461 == 2 ? indices.field_2 : (_temp_var_1461 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1460 == 0 ? indices.field_0 : (_temp_var_1460 == 1 ? indices.field_1 : (_temp_var_1460 == 2 ? indices.field_2 : (_temp_var_1460 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_977_ is already defined
#ifndef _block_k_977__func
#define _block_k_977__func
__device__ int _block_k_977_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1463 = ((({ int _temp_var_1464 = ((({ int _temp_var_1465 = ((i % 4));
        (_temp_var_1465 == 0 ? indices.field_0 : (_temp_var_1465 == 1 ? indices.field_1 : (_temp_var_1465 == 2 ? indices.field_2 : (_temp_var_1465 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1464 == 0 ? indices.field_0 : (_temp_var_1464 == 1 ? indices.field_1 : (_temp_var_1464 == 2 ? indices.field_2 : (_temp_var_1464 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1463 == 0 ? indices.field_0 : (_temp_var_1463 == 1 ? indices.field_1 : (_temp_var_1463 == 2 ? indices.field_2 : (_temp_var_1463 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_979_ is already defined
#ifndef _block_k_979__func
#define _block_k_979__func
__device__ int _block_k_979_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1466 = ((({ int _temp_var_1467 = ((({ int _temp_var_1468 = ((i % 4));
        (_temp_var_1468 == 0 ? indices.field_0 : (_temp_var_1468 == 1 ? indices.field_1 : (_temp_var_1468 == 2 ? indices.field_2 : (_temp_var_1468 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1467 == 0 ? indices.field_0 : (_temp_var_1467 == 1 ? indices.field_1 : (_temp_var_1467 == 2 ? indices.field_2 : (_temp_var_1467 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1466 == 0 ? indices.field_0 : (_temp_var_1466 == 1 ? indices.field_1 : (_temp_var_1466 == 2 ? indices.field_2 : (_temp_var_1466 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_981_ is already defined
#ifndef _block_k_981__func
#define _block_k_981__func
__device__ int _block_k_981_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1469 = ((({ int _temp_var_1470 = ((({ int _temp_var_1471 = ((i % 4));
        (_temp_var_1471 == 0 ? indices.field_0 : (_temp_var_1471 == 1 ? indices.field_1 : (_temp_var_1471 == 2 ? indices.field_2 : (_temp_var_1471 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1470 == 0 ? indices.field_0 : (_temp_var_1470 == 1 ? indices.field_1 : (_temp_var_1470 == 2 ? indices.field_2 : (_temp_var_1470 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1469 == 0 ? indices.field_0 : (_temp_var_1469 == 1 ? indices.field_1 : (_temp_var_1469 == 2 ? indices.field_2 : (_temp_var_1469 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_983_ is already defined
#ifndef _block_k_983__func
#define _block_k_983__func
__device__ int _block_k_983_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1472 = ((({ int _temp_var_1473 = ((({ int _temp_var_1474 = ((i % 4));
        (_temp_var_1474 == 0 ? indices.field_0 : (_temp_var_1474 == 1 ? indices.field_1 : (_temp_var_1474 == 2 ? indices.field_2 : (_temp_var_1474 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1473 == 0 ? indices.field_0 : (_temp_var_1473 == 1 ? indices.field_1 : (_temp_var_1473 == 2 ? indices.field_2 : (_temp_var_1473 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1472 == 0 ? indices.field_0 : (_temp_var_1472 == 1 ? indices.field_1 : (_temp_var_1472 == 2 ? indices.field_2 : (_temp_var_1472 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_985_ is already defined
#ifndef _block_k_985__func
#define _block_k_985__func
__device__ int _block_k_985_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1475 = ((({ int _temp_var_1476 = ((({ int _temp_var_1477 = ((i % 4));
        (_temp_var_1477 == 0 ? indices.field_0 : (_temp_var_1477 == 1 ? indices.field_1 : (_temp_var_1477 == 2 ? indices.field_2 : (_temp_var_1477 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1476 == 0 ? indices.field_0 : (_temp_var_1476 == 1 ? indices.field_1 : (_temp_var_1476 == 2 ? indices.field_2 : (_temp_var_1476 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1475 == 0 ? indices.field_0 : (_temp_var_1475 == 1 ? indices.field_1 : (_temp_var_1475 == 2 ? indices.field_2 : (_temp_var_1475 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_987_ is already defined
#ifndef _block_k_987__func
#define _block_k_987__func
__device__ int _block_k_987_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1478 = ((({ int _temp_var_1479 = ((({ int _temp_var_1480 = ((i % 4));
        (_temp_var_1480 == 0 ? indices.field_0 : (_temp_var_1480 == 1 ? indices.field_1 : (_temp_var_1480 == 2 ? indices.field_2 : (_temp_var_1480 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1479 == 0 ? indices.field_0 : (_temp_var_1479 == 1 ? indices.field_1 : (_temp_var_1479 == 2 ? indices.field_2 : (_temp_var_1479 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1478 == 0 ? indices.field_0 : (_temp_var_1478 == 1 ? indices.field_1 : (_temp_var_1478 == 2 ? indices.field_2 : (_temp_var_1478 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_989_ is already defined
#ifndef _block_k_989__func
#define _block_k_989__func
__device__ int _block_k_989_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1481 = ((({ int _temp_var_1482 = ((({ int _temp_var_1483 = ((i % 4));
        (_temp_var_1483 == 0 ? indices.field_0 : (_temp_var_1483 == 1 ? indices.field_1 : (_temp_var_1483 == 2 ? indices.field_2 : (_temp_var_1483 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1482 == 0 ? indices.field_0 : (_temp_var_1482 == 1 ? indices.field_1 : (_temp_var_1482 == 2 ? indices.field_2 : (_temp_var_1482 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1481 == 0 ? indices.field_0 : (_temp_var_1481 == 1 ? indices.field_1 : (_temp_var_1481 == 2 ? indices.field_2 : (_temp_var_1481 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_991_ is already defined
#ifndef _block_k_991__func
#define _block_k_991__func
__device__ int _block_k_991_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1484 = ((({ int _temp_var_1485 = ((({ int _temp_var_1486 = ((i % 4));
        (_temp_var_1486 == 0 ? indices.field_0 : (_temp_var_1486 == 1 ? indices.field_1 : (_temp_var_1486 == 2 ? indices.field_2 : (_temp_var_1486 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1485 == 0 ? indices.field_0 : (_temp_var_1485 == 1 ? indices.field_1 : (_temp_var_1485 == 2 ? indices.field_2 : (_temp_var_1485 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1484 == 0 ? indices.field_0 : (_temp_var_1484 == 1 ? indices.field_1 : (_temp_var_1484 == 2 ? indices.field_2 : (_temp_var_1484 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_993_ is already defined
#ifndef _block_k_993__func
#define _block_k_993__func
__device__ int _block_k_993_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1487 = ((({ int _temp_var_1488 = ((({ int _temp_var_1489 = ((i % 4));
        (_temp_var_1489 == 0 ? indices.field_0 : (_temp_var_1489 == 1 ? indices.field_1 : (_temp_var_1489 == 2 ? indices.field_2 : (_temp_var_1489 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1488 == 0 ? indices.field_0 : (_temp_var_1488 == 1 ? indices.field_1 : (_temp_var_1488 == 2 ? indices.field_2 : (_temp_var_1488 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1487 == 0 ? indices.field_0 : (_temp_var_1487 == 1 ? indices.field_1 : (_temp_var_1487 == 2 ? indices.field_2 : (_temp_var_1487 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_995_ is already defined
#ifndef _block_k_995__func
#define _block_k_995__func
__device__ int _block_k_995_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1490 = ((({ int _temp_var_1491 = ((({ int _temp_var_1492 = ((i % 4));
        (_temp_var_1492 == 0 ? indices.field_0 : (_temp_var_1492 == 1 ? indices.field_1 : (_temp_var_1492 == 2 ? indices.field_2 : (_temp_var_1492 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1491 == 0 ? indices.field_0 : (_temp_var_1491 == 1 ? indices.field_1 : (_temp_var_1491 == 2 ? indices.field_2 : (_temp_var_1491 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1490 == 0 ? indices.field_0 : (_temp_var_1490 == 1 ? indices.field_1 : (_temp_var_1490 == 2 ? indices.field_2 : (_temp_var_1490 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_997_ is already defined
#ifndef _block_k_997__func
#define _block_k_997__func
__device__ int _block_k_997_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1493 = ((({ int _temp_var_1494 = ((({ int _temp_var_1495 = ((i % 4));
        (_temp_var_1495 == 0 ? indices.field_0 : (_temp_var_1495 == 1 ? indices.field_1 : (_temp_var_1495 == 2 ? indices.field_2 : (_temp_var_1495 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1494 == 0 ? indices.field_0 : (_temp_var_1494 == 1 ? indices.field_1 : (_temp_var_1494 == 2 ? indices.field_2 : (_temp_var_1494 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1493 == 0 ? indices.field_0 : (_temp_var_1493 == 1 ? indices.field_1 : (_temp_var_1493 == 2 ? indices.field_2 : (_temp_var_1493 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_999_ is already defined
#ifndef _block_k_999__func
#define _block_k_999__func
__device__ int _block_k_999_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1496 = ((({ int _temp_var_1497 = ((({ int _temp_var_1498 = ((i % 4));
        (_temp_var_1498 == 0 ? indices.field_0 : (_temp_var_1498 == 1 ? indices.field_1 : (_temp_var_1498 == 2 ? indices.field_2 : (_temp_var_1498 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1497 == 0 ? indices.field_0 : (_temp_var_1497 == 1 ? indices.field_1 : (_temp_var_1497 == 2 ? indices.field_2 : (_temp_var_1497 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1496 == 0 ? indices.field_0 : (_temp_var_1496 == 1 ? indices.field_1 : (_temp_var_1496 == 2 ? indices.field_2 : (_temp_var_1496 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif



// TODO: There should be a better to check if _block_k_1001_ is already defined
#ifndef _block_k_1001__func
#define _block_k_1001__func
__device__ int _block_k_1001_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((((((i % 938)) + ((i / 97)))) % 97717)) + ((((({ int _temp_var_1499 = ((({ int _temp_var_1500 = ((({ int _temp_var_1501 = ((i % 4));
        (_temp_var_1501 == 0 ? indices.field_0 : (_temp_var_1501 == 1 ? indices.field_1 : (_temp_var_1501 == 2 ? indices.field_2 : (_temp_var_1501 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1500 == 0 ? indices.field_0 : (_temp_var_1500 == 1 ? indices.field_1 : (_temp_var_1500 == 2 ? indices.field_2 : (_temp_var_1500 == 3 ? indices.field_3 : NULL)))); }) % 4));
        (_temp_var_1499 == 0 ? indices.field_0 : (_temp_var_1499 == 1 ? indices.field_1 : (_temp_var_1499 == 2 ? indices.field_2 : (_temp_var_1499 == 3 ? indices.field_3 : NULL)))); }) * ((i % 7)))) % 99)));
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_1001_(_env_, _block_k_999_(_env_, _block_k_997_(_env_, _block_k_995_(_env_, _block_k_993_(_env_, _block_k_991_(_env_, _block_k_989_(_env_, _block_k_987_(_env_, _block_k_985_(_env_, _block_k_983_(_env_, _block_k_981_(_env_, _block_k_979_(_env_, _block_k_977_(_env_, _block_k_975_(_env_, _block_k_973_(_env_, _block_k_971_(_env_, _block_k_969_(_env_, _block_k_967_(_env_, _block_k_965_(_env_, _block_k_963_(_env_, _block_k_961_(_env_, _block_k_959_(_env_, _block_k_957_(_env_, _block_k_955_(_env_, _block_k_953_(_env_, _block_k_951_(_env_, _block_k_949_(_env_, _block_k_947_(_env_, _block_k_945_(_env_, _block_k_943_(_env_, _block_k_941_(_env_, _block_k_939_(_env_, _block_k_937_(_env_, _block_k_935_(_env_, _block_k_933_(_env_, _block_k_931_(_env_, _block_k_929_(_env_, _block_k_927_(_env_, _block_k_925_(_env_, _block_k_923_(_env_, _block_k_921_(_env_, _block_k_919_(_env_, _block_k_917_(_env_, _block_k_915_(_env_, _block_k_913_(_env_, _block_k_911_(_env_, _block_k_909_(_env_, _block_k_907_(_env_, _block_k_905_(_env_, _block_k_903_(_env_, _block_k_901_(_env_, _block_k_899_(_env_, _block_k_897_(_env_, _block_k_895_(_env_, _block_k_893_(_env_, _block_k_891_(_env_, _block_k_889_(_env_, _block_k_887_(_env_, _block_k_885_(_env_, _block_k_883_(_env_, _block_k_881_(_env_, _block_k_879_(_env_, _block_k_877_(_env_, _block_k_875_(_env_, _block_k_873_(_env_, _block_k_871_(_env_, _block_k_869_(_env_, _block_k_867_(_env_, _block_k_865_(_env_, _block_k_863_(_env_, _block_k_861_(_env_, _block_k_859_(_env_, _block_k_857_(_env_, _block_k_855_(_env_, _block_k_853_(_env_, _block_k_851_(_env_, _block_k_849_(_env_, _block_k_847_(_env_, _block_k_845_(_env_, _block_k_843_(_env_, _block_k_841_(_env_, _block_k_839_(_env_, _block_k_837_(_env_, _block_k_835_(_env_, _block_k_833_(_env_, _block_k_831_(_env_, _block_k_829_(_env_, _block_k_827_(_env_, _block_k_825_(_env_, _block_k_823_(_env_, _block_k_821_(_env_, _block_k_819_(_env_, _block_k_817_(_env_, _block_k_815_(_env_, _block_k_813_(_env_, _block_k_811_(_env_, _block_k_809_(_env_, _block_k_807_(_env_, _block_k_805_(_env_, _block_k_803_(_env_, _block_k_801_(_env_, _block_k_799_(_env_, _block_k_797_(_env_, _block_k_795_(_env_, _block_k_793_(_env_, _block_k_791_(_env_, _block_k_789_(_env_, _block_k_787_(_env_, _block_k_785_(_env_, _block_k_783_(_env_, _block_k_781_(_env_, _block_k_779_(_env_, _block_k_777_(_env_, _block_k_775_(_env_, _block_k_773_(_env_, _block_k_771_(_env_, _block_k_769_(_env_, _block_k_767_(_env_, _block_k_765_(_env_, _block_k_763_(_env_, _block_k_761_(_env_, _block_k_759_(_env_, _block_k_757_(_env_, _block_k_755_(_env_, _block_k_753_(_env_, _block_k_751_(_env_, _block_k_749_(_env_, _block_k_747_(_env_, _block_k_745_(_env_, _block_k_743_(_env_, _block_k_741_(_env_, _block_k_739_(_env_, _block_k_737_(_env_, _block_k_735_(_env_, _block_k_733_(_env_, _block_k_731_(_env_, _block_k_729_(_env_, _block_k_727_(_env_, _block_k_725_(_env_, _block_k_723_(_env_, _block_k_721_(_env_, _block_k_719_(_env_, _block_k_717_(_env_, _block_k_715_(_env_, _block_k_713_(_env_, _block_k_711_(_env_, _block_k_709_(_env_, _block_k_707_(_env_, _block_k_705_(_env_, _block_k_703_(_env_, _block_k_701_(_env_, _block_k_699_(_env_, _block_k_697_(_env_, _block_k_695_(_env_, _block_k_693_(_env_, _block_k_691_(_env_, _block_k_689_(_env_, _block_k_687_(_env_, _block_k_685_(_env_, _block_k_683_(_env_, _block_k_681_(_env_, _block_k_679_(_env_, _block_k_677_(_env_, _block_k_675_(_env_, _block_k_673_(_env_, _block_k_671_(_env_, _block_k_669_(_env_, _block_k_667_(_env_, _block_k_665_(_env_, _block_k_663_(_env_, _block_k_661_(_env_, _block_k_659_(_env_, _block_k_657_(_env_, _block_k_655_(_env_, _block_k_653_(_env_, _block_k_651_(_env_, _block_k_649_(_env_, _block_k_647_(_env_, _block_k_645_(_env_, _block_k_643_(_env_, _block_k_641_(_env_, _block_k_639_(_env_, _block_k_637_(_env_, _block_k_635_(_env_, _block_k_633_(_env_, _block_k_631_(_env_, _block_k_629_(_env_, _block_k_627_(_env_, _block_k_625_(_env_, _block_k_623_(_env_, _block_k_621_(_env_, _block_k_619_(_env_, _block_k_617_(_env_, _block_k_615_(_env_, _block_k_613_(_env_, _block_k_611_(_env_, _block_k_609_(_env_, _block_k_607_(_env_, _block_k_605_(_env_, _block_k_603_(_env_, _block_k_601_(_env_, _block_k_599_(_env_, _block_k_597_(_env_, _block_k_595_(_env_, _block_k_593_(_env_, _block_k_591_(_env_, _block_k_589_(_env_, _block_k_587_(_env_, _block_k_585_(_env_, _block_k_583_(_env_, _block_k_581_(_env_, _block_k_579_(_env_, _block_k_577_(_env_, _block_k_575_(_env_, _block_k_573_(_env_, _block_k_571_(_env_, _block_k_569_(_env_, _block_k_567_(_env_, _block_k_565_(_env_, _block_k_563_(_env_, _block_k_561_(_env_, _block_k_559_(_env_, _block_k_557_(_env_, _block_k_555_(_env_, _block_k_553_(_env_, _block_k_551_(_env_, _block_k_549_(_env_, _block_k_547_(_env_, _block_k_545_(_env_, _block_k_543_(_env_, _block_k_541_(_env_, _block_k_539_(_env_, _block_k_537_(_env_, _block_k_535_(_env_, _block_k_533_(_env_, _block_k_531_(_env_, _block_k_529_(_env_, _block_k_527_(_env_, _block_k_525_(_env_, _block_k_523_(_env_, _block_k_521_(_env_, _block_k_519_(_env_, _block_k_517_(_env_, _block_k_515_(_env_, _block_k_513_(_env_, _block_k_511_(_env_, _block_k_509_(_env_, _block_k_507_(_env_, _block_k_505_(_env_, _block_k_503_(_env_, _block_k_501_(_env_, _block_k_499_(_env_, _block_k_497_(_env_, _block_k_495_(_env_, _block_k_493_(_env_, _block_k_491_(_env_, _block_k_489_(_env_, _block_k_487_(_env_, _block_k_485_(_env_, _block_k_483_(_env_, _block_k_481_(_env_, _block_k_479_(_env_, _block_k_477_(_env_, _block_k_475_(_env_, _block_k_473_(_env_, _block_k_471_(_env_, _block_k_469_(_env_, _block_k_467_(_env_, _block_k_465_(_env_, _block_k_463_(_env_, _block_k_461_(_env_, _block_k_459_(_env_, _block_k_457_(_env_, _block_k_455_(_env_, _block_k_453_(_env_, _block_k_451_(_env_, _block_k_449_(_env_, _block_k_447_(_env_, _block_k_445_(_env_, _block_k_443_(_env_, _block_k_441_(_env_, _block_k_439_(_env_, _block_k_437_(_env_, _block_k_435_(_env_, _block_k_433_(_env_, _block_k_431_(_env_, _block_k_429_(_env_, _block_k_427_(_env_, _block_k_425_(_env_, _block_k_423_(_env_, _block_k_421_(_env_, _block_k_419_(_env_, _block_k_417_(_env_, _block_k_415_(_env_, _block_k_413_(_env_, _block_k_411_(_env_, _block_k_409_(_env_, _block_k_407_(_env_, _block_k_405_(_env_, _block_k_403_(_env_, _block_k_401_(_env_, _block_k_399_(_env_, _block_k_397_(_env_, _block_k_395_(_env_, _block_k_393_(_env_, _block_k_391_(_env_, _block_k_389_(_env_, _block_k_387_(_env_, _block_k_385_(_env_, _block_k_383_(_env_, _block_k_381_(_env_, _block_k_379_(_env_, _block_k_377_(_env_, _block_k_375_(_env_, _block_k_373_(_env_, _block_k_371_(_env_, _block_k_369_(_env_, _block_k_367_(_env_, _block_k_365_(_env_, _block_k_363_(_env_, _block_k_361_(_env_, _block_k_359_(_env_, _block_k_357_(_env_, _block_k_355_(_env_, _block_k_353_(_env_, _block_k_351_(_env_, _block_k_349_(_env_, _block_k_347_(_env_, _block_k_345_(_env_, _block_k_343_(_env_, _block_k_341_(_env_, _block_k_339_(_env_, _block_k_337_(_env_, _block_k_335_(_env_, _block_k_333_(_env_, _block_k_331_(_env_, _block_k_329_(_env_, _block_k_327_(_env_, _block_k_325_(_env_, _block_k_323_(_env_, _block_k_321_(_env_, _block_k_319_(_env_, _block_k_317_(_env_, _block_k_315_(_env_, _block_k_313_(_env_, _block_k_311_(_env_, _block_k_309_(_env_, _block_k_307_(_env_, _block_k_305_(_env_, _block_k_303_(_env_, _block_k_301_(_env_, _block_k_299_(_env_, _block_k_297_(_env_, _block_k_295_(_env_, _block_k_293_(_env_, _block_k_291_(_env_, _block_k_289_(_env_, _block_k_287_(_env_, _block_k_285_(_env_, _block_k_283_(_env_, _block_k_281_(_env_, _block_k_279_(_env_, _block_k_277_(_env_, _block_k_275_(_env_, _block_k_273_(_env_, _block_k_271_(_env_, _block_k_269_(_env_, _block_k_267_(_env_, _block_k_265_(_env_, _block_k_263_(_env_, _block_k_261_(_env_, _block_k_259_(_env_, _block_k_257_(_env_, _block_k_255_(_env_, _block_k_253_(_env_, _block_k_251_(_env_, _block_k_249_(_env_, _block_k_247_(_env_, _block_k_245_(_env_, _block_k_243_(_env_, _block_k_241_(_env_, _block_k_239_(_env_, _block_k_237_(_env_, _block_k_235_(_env_, _block_k_233_(_env_, _block_k_231_(_env_, _block_k_229_(_env_, _block_k_227_(_env_, _block_k_225_(_env_, _block_k_223_(_env_, _block_k_221_(_env_, _block_k_219_(_env_, _block_k_217_(_env_, _block_k_215_(_env_, _block_k_213_(_env_, _block_k_211_(_env_, _block_k_209_(_env_, _block_k_207_(_env_, _block_k_205_(_env_, _block_k_203_(_env_, _block_k_201_(_env_, _block_k_199_(_env_, _block_k_197_(_env_, _block_k_195_(_env_, _block_k_193_(_env_, _block_k_191_(_env_, _block_k_189_(_env_, _block_k_187_(_env_, _block_k_185_(_env_, _block_k_183_(_env_, _block_k_181_(_env_, _block_k_179_(_env_, _block_k_177_(_env_, _block_k_175_(_env_, _block_k_173_(_env_, _block_k_171_(_env_, _block_k_169_(_env_, _block_k_167_(_env_, _block_k_165_(_env_, _block_k_163_(_env_, _block_k_161_(_env_, _block_k_159_(_env_, _block_k_157_(_env_, _block_k_155_(_env_, _block_k_153_(_env_, _block_k_151_(_env_, _block_k_149_(_env_, _block_k_147_(_env_, _block_k_145_(_env_, _block_k_143_(_env_, _block_k_141_(_env_, _block_k_139_(_env_, _block_k_137_(_env_, _block_k_135_(_env_, _block_k_133_(_env_, _block_k_131_(_env_, _block_k_129_(_env_, _block_k_127_(_env_, _block_k_125_(_env_, _block_k_123_(_env_, _block_k_121_(_env_, _block_k_119_(_env_, _block_k_117_(_env_, _block_k_115_(_env_, _block_k_113_(_env_, _block_k_111_(_env_, _block_k_109_(_env_, _block_k_107_(_env_, _block_k_105_(_env_, _block_k_103_(_env_, _block_k_101_(_env_, _block_k_99_(_env_, _block_k_97_(_env_, _block_k_95_(_env_, _block_k_93_(_env_, _block_k_91_(_env_, _block_k_89_(_env_, _block_k_87_(_env_, _block_k_85_(_env_, _block_k_83_(_env_, _block_k_81_(_env_, _block_k_79_(_env_, _block_k_77_(_env_, _block_k_75_(_env_, _block_k_73_(_env_, _block_k_71_(_env_, _block_k_69_(_env_, _block_k_67_(_env_, _block_k_65_(_env_, _block_k_63_(_env_, _block_k_61_(_env_, _block_k_59_(_env_, _block_k_57_(_env_, _block_k_55_(_env_, _block_k_53_(_env_, _block_k_51_(_env_, _block_k_49_(_env_, _block_k_47_(_env_, _block_k_45_(_env_, _block_k_43_(_env_, _block_k_41_(_env_, _block_k_39_(_env_, _block_k_37_(_env_, _block_k_35_(_env_, _block_k_33_(_env_, _block_k_31_(_env_, _block_k_29_(_env_, _block_k_27_(_env_, _block_k_25_(_env_, _block_k_23_(_env_, _block_k_21_(_env_, _block_k_19_(_env_, _block_k_17_(_env_, _block_k_15_(_env_, _block_k_13_(_env_, _block_k_11_(_env_, _block_k_9_(_env_, _block_k_7_(_env_, _block_k_5_(_env_, _block_k_3_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 500000, (_tid_ / 1000) % 500, (_tid_ / 2) % 500, (_tid_ / 1) % 2}));
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
    int * _kernel_result_2;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(int) * 10000000)));
    program_result->device_allocations->push_back(_kernel_result_2);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_1<<<39063, 256>>>(dev_env, 10000000, _kernel_result_2);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);

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
        timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_2));
    timeReportMeasure(program_result, free_memory);


    delete program_result->device_allocations;
    
    return program_result;
}
