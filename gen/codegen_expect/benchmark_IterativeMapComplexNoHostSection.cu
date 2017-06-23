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
        return (indices.field_2 % 133777);
    }
}

#endif



// TODO: There should be a better to check if _block_k_3_ is already defined
#ifndef _block_k_3__func
#define _block_k_3__func
__device__ int _block_k_3_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_5_ is already defined
#ifndef _block_k_5__func
#define _block_k_5__func
__device__ int _block_k_5_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_7_ is already defined
#ifndef _block_k_7__func
#define _block_k_7__func
__device__ int _block_k_7_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_9_ is already defined
#ifndef _block_k_9__func
#define _block_k_9__func
__device__ int _block_k_9_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_11_ is already defined
#ifndef _block_k_11__func
#define _block_k_11__func
__device__ int _block_k_11_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_13_ is already defined
#ifndef _block_k_13__func
#define _block_k_13__func
__device__ int _block_k_13_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_15_ is already defined
#ifndef _block_k_15__func
#define _block_k_15__func
__device__ int _block_k_15_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_17_ is already defined
#ifndef _block_k_17__func
#define _block_k_17__func
__device__ int _block_k_17_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_19_ is already defined
#ifndef _block_k_19__func
#define _block_k_19__func
__device__ int _block_k_19_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_21_ is already defined
#ifndef _block_k_21__func
#define _block_k_21__func
__device__ int _block_k_21_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
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



// TODO: There should be a better to check if _block_k_25_ is already defined
#ifndef _block_k_25__func
#define _block_k_25__func
__device__ int _block_k_25_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
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



// TODO: There should be a better to check if _block_k_29_ is already defined
#ifndef _block_k_29__func
#define _block_k_29__func
__device__ int _block_k_29_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_31_ is already defined
#ifndef _block_k_31__func
#define _block_k_31__func
__device__ int _block_k_31_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_33_ is already defined
#ifndef _block_k_33__func
#define _block_k_33__func
__device__ int _block_k_33_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_35_ is already defined
#ifndef _block_k_35__func
#define _block_k_35__func
__device__ int _block_k_35_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_37_ is already defined
#ifndef _block_k_37__func
#define _block_k_37__func
__device__ int _block_k_37_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_39_ is already defined
#ifndef _block_k_39__func
#define _block_k_39__func
__device__ int _block_k_39_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_41_ is already defined
#ifndef _block_k_41__func
#define _block_k_41__func
__device__ int _block_k_41_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_43_ is already defined
#ifndef _block_k_43__func
#define _block_k_43__func
__device__ int _block_k_43_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_45_ is already defined
#ifndef _block_k_45__func
#define _block_k_45__func
__device__ int _block_k_45_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_47_ is already defined
#ifndef _block_k_47__func
#define _block_k_47__func
__device__ int _block_k_47_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_49_ is already defined
#ifndef _block_k_49__func
#define _block_k_49__func
__device__ int _block_k_49_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_51_ is already defined
#ifndef _block_k_51__func
#define _block_k_51__func
__device__ int _block_k_51_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_53_ is already defined
#ifndef _block_k_53__func
#define _block_k_53__func
__device__ int _block_k_53_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_55_ is already defined
#ifndef _block_k_55__func
#define _block_k_55__func
__device__ int _block_k_55_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_57_ is already defined
#ifndef _block_k_57__func
#define _block_k_57__func
__device__ int _block_k_57_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_59_ is already defined
#ifndef _block_k_59__func
#define _block_k_59__func
__device__ int _block_k_59_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_61_ is already defined
#ifndef _block_k_61__func
#define _block_k_61__func
__device__ int _block_k_61_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_63_ is already defined
#ifndef _block_k_63__func
#define _block_k_63__func
__device__ int _block_k_63_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_65_ is already defined
#ifndef _block_k_65__func
#define _block_k_65__func
__device__ int _block_k_65_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_67_ is already defined
#ifndef _block_k_67__func
#define _block_k_67__func
__device__ int _block_k_67_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_69_ is already defined
#ifndef _block_k_69__func
#define _block_k_69__func
__device__ int _block_k_69_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_71_ is already defined
#ifndef _block_k_71__func
#define _block_k_71__func
__device__ int _block_k_71_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_73_ is already defined
#ifndef _block_k_73__func
#define _block_k_73__func
__device__ int _block_k_73_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_75_ is already defined
#ifndef _block_k_75__func
#define _block_k_75__func
__device__ int _block_k_75_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_77_ is already defined
#ifndef _block_k_77__func
#define _block_k_77__func
__device__ int _block_k_77_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_79_ is already defined
#ifndef _block_k_79__func
#define _block_k_79__func
__device__ int _block_k_79_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_81_ is already defined
#ifndef _block_k_81__func
#define _block_k_81__func
__device__ int _block_k_81_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_83_ is already defined
#ifndef _block_k_83__func
#define _block_k_83__func
__device__ int _block_k_83_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_85_ is already defined
#ifndef _block_k_85__func
#define _block_k_85__func
__device__ int _block_k_85_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_87_ is already defined
#ifndef _block_k_87__func
#define _block_k_87__func
__device__ int _block_k_87_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_89_ is already defined
#ifndef _block_k_89__func
#define _block_k_89__func
__device__ int _block_k_89_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_91_ is already defined
#ifndef _block_k_91__func
#define _block_k_91__func
__device__ int _block_k_91_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_93_ is already defined
#ifndef _block_k_93__func
#define _block_k_93__func
__device__ int _block_k_93_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_95_ is already defined
#ifndef _block_k_95__func
#define _block_k_95__func
__device__ int _block_k_95_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_97_ is already defined
#ifndef _block_k_97__func
#define _block_k_97__func
__device__ int _block_k_97_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_99_ is already defined
#ifndef _block_k_99__func
#define _block_k_99__func
__device__ int _block_k_99_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_101_ is already defined
#ifndef _block_k_101__func
#define _block_k_101__func
__device__ int _block_k_101_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_103_ is already defined
#ifndef _block_k_103__func
#define _block_k_103__func
__device__ int _block_k_103_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
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



// TODO: There should be a better to check if _block_k_107_ is already defined
#ifndef _block_k_107__func
#define _block_k_107__func
__device__ int _block_k_107_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_109_ is already defined
#ifndef _block_k_109__func
#define _block_k_109__func
__device__ int _block_k_109_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_111_ is already defined
#ifndef _block_k_111__func
#define _block_k_111__func
__device__ int _block_k_111_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_113_ is already defined
#ifndef _block_k_113__func
#define _block_k_113__func
__device__ int _block_k_113_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_115_ is already defined
#ifndef _block_k_115__func
#define _block_k_115__func
__device__ int _block_k_115_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_117_ is already defined
#ifndef _block_k_117__func
#define _block_k_117__func
__device__ int _block_k_117_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_119_ is already defined
#ifndef _block_k_119__func
#define _block_k_119__func
__device__ int _block_k_119_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_121_ is already defined
#ifndef _block_k_121__func
#define _block_k_121__func
__device__ int _block_k_121_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_123_ is already defined
#ifndef _block_k_123__func
#define _block_k_123__func
__device__ int _block_k_123_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_125_ is already defined
#ifndef _block_k_125__func
#define _block_k_125__func
__device__ int _block_k_125_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_127_ is already defined
#ifndef _block_k_127__func
#define _block_k_127__func
__device__ int _block_k_127_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_129_ is already defined
#ifndef _block_k_129__func
#define _block_k_129__func
__device__ int _block_k_129_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_131_ is already defined
#ifndef _block_k_131__func
#define _block_k_131__func
__device__ int _block_k_131_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_133_ is already defined
#ifndef _block_k_133__func
#define _block_k_133__func
__device__ int _block_k_133_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_135_ is already defined
#ifndef _block_k_135__func
#define _block_k_135__func
__device__ int _block_k_135_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_137_ is already defined
#ifndef _block_k_137__func
#define _block_k_137__func
__device__ int _block_k_137_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_139_ is already defined
#ifndef _block_k_139__func
#define _block_k_139__func
__device__ int _block_k_139_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_141_ is already defined
#ifndef _block_k_141__func
#define _block_k_141__func
__device__ int _block_k_141_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_143_ is already defined
#ifndef _block_k_143__func
#define _block_k_143__func
__device__ int _block_k_143_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_145_ is already defined
#ifndef _block_k_145__func
#define _block_k_145__func
__device__ int _block_k_145_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_147_ is already defined
#ifndef _block_k_147__func
#define _block_k_147__func
__device__ int _block_k_147_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_149_ is already defined
#ifndef _block_k_149__func
#define _block_k_149__func
__device__ int _block_k_149_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_151_ is already defined
#ifndef _block_k_151__func
#define _block_k_151__func
__device__ int _block_k_151_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_153_ is already defined
#ifndef _block_k_153__func
#define _block_k_153__func
__device__ int _block_k_153_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_155_ is already defined
#ifndef _block_k_155__func
#define _block_k_155__func
__device__ int _block_k_155_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_157_ is already defined
#ifndef _block_k_157__func
#define _block_k_157__func
__device__ int _block_k_157_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_159_ is already defined
#ifndef _block_k_159__func
#define _block_k_159__func
__device__ int _block_k_159_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_161_ is already defined
#ifndef _block_k_161__func
#define _block_k_161__func
__device__ int _block_k_161_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_163_ is already defined
#ifndef _block_k_163__func
#define _block_k_163__func
__device__ int _block_k_163_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_165_ is already defined
#ifndef _block_k_165__func
#define _block_k_165__func
__device__ int _block_k_165_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_167_ is already defined
#ifndef _block_k_167__func
#define _block_k_167__func
__device__ int _block_k_167_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_169_ is already defined
#ifndef _block_k_169__func
#define _block_k_169__func
__device__ int _block_k_169_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_171_ is already defined
#ifndef _block_k_171__func
#define _block_k_171__func
__device__ int _block_k_171_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_173_ is already defined
#ifndef _block_k_173__func
#define _block_k_173__func
__device__ int _block_k_173_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_175_ is already defined
#ifndef _block_k_175__func
#define _block_k_175__func
__device__ int _block_k_175_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_177_ is already defined
#ifndef _block_k_177__func
#define _block_k_177__func
__device__ int _block_k_177_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_179_ is already defined
#ifndef _block_k_179__func
#define _block_k_179__func
__device__ int _block_k_179_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_181_ is already defined
#ifndef _block_k_181__func
#define _block_k_181__func
__device__ int _block_k_181_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_183_ is already defined
#ifndef _block_k_183__func
#define _block_k_183__func
__device__ int _block_k_183_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_185_ is already defined
#ifndef _block_k_185__func
#define _block_k_185__func
__device__ int _block_k_185_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_187_ is already defined
#ifndef _block_k_187__func
#define _block_k_187__func
__device__ int _block_k_187_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_189_ is already defined
#ifndef _block_k_189__func
#define _block_k_189__func
__device__ int _block_k_189_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_191_ is already defined
#ifndef _block_k_191__func
#define _block_k_191__func
__device__ int _block_k_191_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_193_ is already defined
#ifndef _block_k_193__func
#define _block_k_193__func
__device__ int _block_k_193_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_195_ is already defined
#ifndef _block_k_195__func
#define _block_k_195__func
__device__ int _block_k_195_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_197_ is already defined
#ifndef _block_k_197__func
#define _block_k_197__func
__device__ int _block_k_197_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_199_ is already defined
#ifndef _block_k_199__func
#define _block_k_199__func
__device__ int _block_k_199_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_201_ is already defined
#ifndef _block_k_201__func
#define _block_k_201__func
__device__ int _block_k_201_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_203_ is already defined
#ifndef _block_k_203__func
#define _block_k_203__func
__device__ int _block_k_203_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_205_ is already defined
#ifndef _block_k_205__func
#define _block_k_205__func
__device__ int _block_k_205_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_207_ is already defined
#ifndef _block_k_207__func
#define _block_k_207__func
__device__ int _block_k_207_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_209_ is already defined
#ifndef _block_k_209__func
#define _block_k_209__func
__device__ int _block_k_209_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_211_ is already defined
#ifndef _block_k_211__func
#define _block_k_211__func
__device__ int _block_k_211_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_213_ is already defined
#ifndef _block_k_213__func
#define _block_k_213__func
__device__ int _block_k_213_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_215_ is already defined
#ifndef _block_k_215__func
#define _block_k_215__func
__device__ int _block_k_215_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_217_ is already defined
#ifndef _block_k_217__func
#define _block_k_217__func
__device__ int _block_k_217_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_219_ is already defined
#ifndef _block_k_219__func
#define _block_k_219__func
__device__ int _block_k_219_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_221_ is already defined
#ifndef _block_k_221__func
#define _block_k_221__func
__device__ int _block_k_221_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_223_ is already defined
#ifndef _block_k_223__func
#define _block_k_223__func
__device__ int _block_k_223_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_225_ is already defined
#ifndef _block_k_225__func
#define _block_k_225__func
__device__ int _block_k_225_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_227_ is already defined
#ifndef _block_k_227__func
#define _block_k_227__func
__device__ int _block_k_227_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_229_ is already defined
#ifndef _block_k_229__func
#define _block_k_229__func
__device__ int _block_k_229_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_231_ is already defined
#ifndef _block_k_231__func
#define _block_k_231__func
__device__ int _block_k_231_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_233_ is already defined
#ifndef _block_k_233__func
#define _block_k_233__func
__device__ int _block_k_233_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_235_ is already defined
#ifndef _block_k_235__func
#define _block_k_235__func
__device__ int _block_k_235_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_237_ is already defined
#ifndef _block_k_237__func
#define _block_k_237__func
__device__ int _block_k_237_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_239_ is already defined
#ifndef _block_k_239__func
#define _block_k_239__func
__device__ int _block_k_239_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_241_ is already defined
#ifndef _block_k_241__func
#define _block_k_241__func
__device__ int _block_k_241_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_243_ is already defined
#ifndef _block_k_243__func
#define _block_k_243__func
__device__ int _block_k_243_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_245_ is already defined
#ifndef _block_k_245__func
#define _block_k_245__func
__device__ int _block_k_245_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_247_ is already defined
#ifndef _block_k_247__func
#define _block_k_247__func
__device__ int _block_k_247_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_249_ is already defined
#ifndef _block_k_249__func
#define _block_k_249__func
__device__ int _block_k_249_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_251_ is already defined
#ifndef _block_k_251__func
#define _block_k_251__func
__device__ int _block_k_251_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_253_ is already defined
#ifndef _block_k_253__func
#define _block_k_253__func
__device__ int _block_k_253_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_255_ is already defined
#ifndef _block_k_255__func
#define _block_k_255__func
__device__ int _block_k_255_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_257_ is already defined
#ifndef _block_k_257__func
#define _block_k_257__func
__device__ int _block_k_257_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_259_ is already defined
#ifndef _block_k_259__func
#define _block_k_259__func
__device__ int _block_k_259_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_261_ is already defined
#ifndef _block_k_261__func
#define _block_k_261__func
__device__ int _block_k_261_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_263_ is already defined
#ifndef _block_k_263__func
#define _block_k_263__func
__device__ int _block_k_263_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_265_ is already defined
#ifndef _block_k_265__func
#define _block_k_265__func
__device__ int _block_k_265_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_267_ is already defined
#ifndef _block_k_267__func
#define _block_k_267__func
__device__ int _block_k_267_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_269_ is already defined
#ifndef _block_k_269__func
#define _block_k_269__func
__device__ int _block_k_269_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_271_ is already defined
#ifndef _block_k_271__func
#define _block_k_271__func
__device__ int _block_k_271_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_273_ is already defined
#ifndef _block_k_273__func
#define _block_k_273__func
__device__ int _block_k_273_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_275_ is already defined
#ifndef _block_k_275__func
#define _block_k_275__func
__device__ int _block_k_275_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_277_ is already defined
#ifndef _block_k_277__func
#define _block_k_277__func
__device__ int _block_k_277_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_279_ is already defined
#ifndef _block_k_279__func
#define _block_k_279__func
__device__ int _block_k_279_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_281_ is already defined
#ifndef _block_k_281__func
#define _block_k_281__func
__device__ int _block_k_281_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_283_ is already defined
#ifndef _block_k_283__func
#define _block_k_283__func
__device__ int _block_k_283_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_285_ is already defined
#ifndef _block_k_285__func
#define _block_k_285__func
__device__ int _block_k_285_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_287_ is already defined
#ifndef _block_k_287__func
#define _block_k_287__func
__device__ int _block_k_287_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_289_ is already defined
#ifndef _block_k_289__func
#define _block_k_289__func
__device__ int _block_k_289_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_291_ is already defined
#ifndef _block_k_291__func
#define _block_k_291__func
__device__ int _block_k_291_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_293_ is already defined
#ifndef _block_k_293__func
#define _block_k_293__func
__device__ int _block_k_293_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_295_ is already defined
#ifndef _block_k_295__func
#define _block_k_295__func
__device__ int _block_k_295_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_297_ is already defined
#ifndef _block_k_297__func
#define _block_k_297__func
__device__ int _block_k_297_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_299_ is already defined
#ifndef _block_k_299__func
#define _block_k_299__func
__device__ int _block_k_299_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_301_ is already defined
#ifndef _block_k_301__func
#define _block_k_301__func
__device__ int _block_k_301_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_303_ is already defined
#ifndef _block_k_303__func
#define _block_k_303__func
__device__ int _block_k_303_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_305_ is already defined
#ifndef _block_k_305__func
#define _block_k_305__func
__device__ int _block_k_305_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_307_ is already defined
#ifndef _block_k_307__func
#define _block_k_307__func
__device__ int _block_k_307_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_309_ is already defined
#ifndef _block_k_309__func
#define _block_k_309__func
__device__ int _block_k_309_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_311_ is already defined
#ifndef _block_k_311__func
#define _block_k_311__func
__device__ int _block_k_311_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_313_ is already defined
#ifndef _block_k_313__func
#define _block_k_313__func
__device__ int _block_k_313_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_315_ is already defined
#ifndef _block_k_315__func
#define _block_k_315__func
__device__ int _block_k_315_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_317_ is already defined
#ifndef _block_k_317__func
#define _block_k_317__func
__device__ int _block_k_317_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_319_ is already defined
#ifndef _block_k_319__func
#define _block_k_319__func
__device__ int _block_k_319_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_321_ is already defined
#ifndef _block_k_321__func
#define _block_k_321__func
__device__ int _block_k_321_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_323_ is already defined
#ifndef _block_k_323__func
#define _block_k_323__func
__device__ int _block_k_323_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_325_ is already defined
#ifndef _block_k_325__func
#define _block_k_325__func
__device__ int _block_k_325_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_327_ is already defined
#ifndef _block_k_327__func
#define _block_k_327__func
__device__ int _block_k_327_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_329_ is already defined
#ifndef _block_k_329__func
#define _block_k_329__func
__device__ int _block_k_329_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_331_ is already defined
#ifndef _block_k_331__func
#define _block_k_331__func
__device__ int _block_k_331_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_333_ is already defined
#ifndef _block_k_333__func
#define _block_k_333__func
__device__ int _block_k_333_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_335_ is already defined
#ifndef _block_k_335__func
#define _block_k_335__func
__device__ int _block_k_335_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_337_ is already defined
#ifndef _block_k_337__func
#define _block_k_337__func
__device__ int _block_k_337_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_339_ is already defined
#ifndef _block_k_339__func
#define _block_k_339__func
__device__ int _block_k_339_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_341_ is already defined
#ifndef _block_k_341__func
#define _block_k_341__func
__device__ int _block_k_341_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_343_ is already defined
#ifndef _block_k_343__func
#define _block_k_343__func
__device__ int _block_k_343_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_345_ is already defined
#ifndef _block_k_345__func
#define _block_k_345__func
__device__ int _block_k_345_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_347_ is already defined
#ifndef _block_k_347__func
#define _block_k_347__func
__device__ int _block_k_347_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_349_ is already defined
#ifndef _block_k_349__func
#define _block_k_349__func
__device__ int _block_k_349_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_351_ is already defined
#ifndef _block_k_351__func
#define _block_k_351__func
__device__ int _block_k_351_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_353_ is already defined
#ifndef _block_k_353__func
#define _block_k_353__func
__device__ int _block_k_353_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_355_ is already defined
#ifndef _block_k_355__func
#define _block_k_355__func
__device__ int _block_k_355_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_357_ is already defined
#ifndef _block_k_357__func
#define _block_k_357__func
__device__ int _block_k_357_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_359_ is already defined
#ifndef _block_k_359__func
#define _block_k_359__func
__device__ int _block_k_359_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_361_ is already defined
#ifndef _block_k_361__func
#define _block_k_361__func
__device__ int _block_k_361_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_363_ is already defined
#ifndef _block_k_363__func
#define _block_k_363__func
__device__ int _block_k_363_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_365_ is already defined
#ifndef _block_k_365__func
#define _block_k_365__func
__device__ int _block_k_365_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_367_ is already defined
#ifndef _block_k_367__func
#define _block_k_367__func
__device__ int _block_k_367_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_369_ is already defined
#ifndef _block_k_369__func
#define _block_k_369__func
__device__ int _block_k_369_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_371_ is already defined
#ifndef _block_k_371__func
#define _block_k_371__func
__device__ int _block_k_371_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_373_ is already defined
#ifndef _block_k_373__func
#define _block_k_373__func
__device__ int _block_k_373_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_375_ is already defined
#ifndef _block_k_375__func
#define _block_k_375__func
__device__ int _block_k_375_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_377_ is already defined
#ifndef _block_k_377__func
#define _block_k_377__func
__device__ int _block_k_377_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_379_ is already defined
#ifndef _block_k_379__func
#define _block_k_379__func
__device__ int _block_k_379_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_381_ is already defined
#ifndef _block_k_381__func
#define _block_k_381__func
__device__ int _block_k_381_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_383_ is already defined
#ifndef _block_k_383__func
#define _block_k_383__func
__device__ int _block_k_383_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_385_ is already defined
#ifndef _block_k_385__func
#define _block_k_385__func
__device__ int _block_k_385_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_387_ is already defined
#ifndef _block_k_387__func
#define _block_k_387__func
__device__ int _block_k_387_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_389_ is already defined
#ifndef _block_k_389__func
#define _block_k_389__func
__device__ int _block_k_389_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_391_ is already defined
#ifndef _block_k_391__func
#define _block_k_391__func
__device__ int _block_k_391_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_393_ is already defined
#ifndef _block_k_393__func
#define _block_k_393__func
__device__ int _block_k_393_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_395_ is already defined
#ifndef _block_k_395__func
#define _block_k_395__func
__device__ int _block_k_395_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_397_ is already defined
#ifndef _block_k_397__func
#define _block_k_397__func
__device__ int _block_k_397_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_399_ is already defined
#ifndef _block_k_399__func
#define _block_k_399__func
__device__ int _block_k_399_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_401_ is already defined
#ifndef _block_k_401__func
#define _block_k_401__func
__device__ int _block_k_401_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_403_ is already defined
#ifndef _block_k_403__func
#define _block_k_403__func
__device__ int _block_k_403_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_405_ is already defined
#ifndef _block_k_405__func
#define _block_k_405__func
__device__ int _block_k_405_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_407_ is already defined
#ifndef _block_k_407__func
#define _block_k_407__func
__device__ int _block_k_407_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_409_ is already defined
#ifndef _block_k_409__func
#define _block_k_409__func
__device__ int _block_k_409_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_411_ is already defined
#ifndef _block_k_411__func
#define _block_k_411__func
__device__ int _block_k_411_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_413_ is already defined
#ifndef _block_k_413__func
#define _block_k_413__func
__device__ int _block_k_413_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_415_ is already defined
#ifndef _block_k_415__func
#define _block_k_415__func
__device__ int _block_k_415_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_417_ is already defined
#ifndef _block_k_417__func
#define _block_k_417__func
__device__ int _block_k_417_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_419_ is already defined
#ifndef _block_k_419__func
#define _block_k_419__func
__device__ int _block_k_419_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_421_ is already defined
#ifndef _block_k_421__func
#define _block_k_421__func
__device__ int _block_k_421_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_423_ is already defined
#ifndef _block_k_423__func
#define _block_k_423__func
__device__ int _block_k_423_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_425_ is already defined
#ifndef _block_k_425__func
#define _block_k_425__func
__device__ int _block_k_425_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_427_ is already defined
#ifndef _block_k_427__func
#define _block_k_427__func
__device__ int _block_k_427_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_429_ is already defined
#ifndef _block_k_429__func
#define _block_k_429__func
__device__ int _block_k_429_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_431_ is already defined
#ifndef _block_k_431__func
#define _block_k_431__func
__device__ int _block_k_431_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_433_ is already defined
#ifndef _block_k_433__func
#define _block_k_433__func
__device__ int _block_k_433_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_435_ is already defined
#ifndef _block_k_435__func
#define _block_k_435__func
__device__ int _block_k_435_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_437_ is already defined
#ifndef _block_k_437__func
#define _block_k_437__func
__device__ int _block_k_437_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_439_ is already defined
#ifndef _block_k_439__func
#define _block_k_439__func
__device__ int _block_k_439_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_441_ is already defined
#ifndef _block_k_441__func
#define _block_k_441__func
__device__ int _block_k_441_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_443_ is already defined
#ifndef _block_k_443__func
#define _block_k_443__func
__device__ int _block_k_443_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_445_ is already defined
#ifndef _block_k_445__func
#define _block_k_445__func
__device__ int _block_k_445_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_447_ is already defined
#ifndef _block_k_447__func
#define _block_k_447__func
__device__ int _block_k_447_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_449_ is already defined
#ifndef _block_k_449__func
#define _block_k_449__func
__device__ int _block_k_449_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_451_ is already defined
#ifndef _block_k_451__func
#define _block_k_451__func
__device__ int _block_k_451_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_453_ is already defined
#ifndef _block_k_453__func
#define _block_k_453__func
__device__ int _block_k_453_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_455_ is already defined
#ifndef _block_k_455__func
#define _block_k_455__func
__device__ int _block_k_455_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_457_ is already defined
#ifndef _block_k_457__func
#define _block_k_457__func
__device__ int _block_k_457_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_459_ is already defined
#ifndef _block_k_459__func
#define _block_k_459__func
__device__ int _block_k_459_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_461_ is already defined
#ifndef _block_k_461__func
#define _block_k_461__func
__device__ int _block_k_461_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_463_ is already defined
#ifndef _block_k_463__func
#define _block_k_463__func
__device__ int _block_k_463_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_465_ is already defined
#ifndef _block_k_465__func
#define _block_k_465__func
__device__ int _block_k_465_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_467_ is already defined
#ifndef _block_k_467__func
#define _block_k_467__func
__device__ int _block_k_467_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_469_ is already defined
#ifndef _block_k_469__func
#define _block_k_469__func
__device__ int _block_k_469_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_471_ is already defined
#ifndef _block_k_471__func
#define _block_k_471__func
__device__ int _block_k_471_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_473_ is already defined
#ifndef _block_k_473__func
#define _block_k_473__func
__device__ int _block_k_473_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_475_ is already defined
#ifndef _block_k_475__func
#define _block_k_475__func
__device__ int _block_k_475_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_477_ is already defined
#ifndef _block_k_477__func
#define _block_k_477__func
__device__ int _block_k_477_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_479_ is already defined
#ifndef _block_k_479__func
#define _block_k_479__func
__device__ int _block_k_479_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_481_ is already defined
#ifndef _block_k_481__func
#define _block_k_481__func
__device__ int _block_k_481_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_483_ is already defined
#ifndef _block_k_483__func
#define _block_k_483__func
__device__ int _block_k_483_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_485_ is already defined
#ifndef _block_k_485__func
#define _block_k_485__func
__device__ int _block_k_485_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_487_ is already defined
#ifndef _block_k_487__func
#define _block_k_487__func
__device__ int _block_k_487_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_489_ is already defined
#ifndef _block_k_489__func
#define _block_k_489__func
__device__ int _block_k_489_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_491_ is already defined
#ifndef _block_k_491__func
#define _block_k_491__func
__device__ int _block_k_491_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_493_ is already defined
#ifndef _block_k_493__func
#define _block_k_493__func
__device__ int _block_k_493_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_495_ is already defined
#ifndef _block_k_495__func
#define _block_k_495__func
__device__ int _block_k_495_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_497_ is already defined
#ifndef _block_k_497__func
#define _block_k_497__func
__device__ int _block_k_497_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_499_ is already defined
#ifndef _block_k_499__func
#define _block_k_499__func
__device__ int _block_k_499_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_501_ is already defined
#ifndef _block_k_501__func
#define _block_k_501__func
__device__ int _block_k_501_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_503_ is already defined
#ifndef _block_k_503__func
#define _block_k_503__func
__device__ int _block_k_503_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_505_ is already defined
#ifndef _block_k_505__func
#define _block_k_505__func
__device__ int _block_k_505_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_507_ is already defined
#ifndef _block_k_507__func
#define _block_k_507__func
__device__ int _block_k_507_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_509_ is already defined
#ifndef _block_k_509__func
#define _block_k_509__func
__device__ int _block_k_509_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_511_ is already defined
#ifndef _block_k_511__func
#define _block_k_511__func
__device__ int _block_k_511_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_513_ is already defined
#ifndef _block_k_513__func
#define _block_k_513__func
__device__ int _block_k_513_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_515_ is already defined
#ifndef _block_k_515__func
#define _block_k_515__func
__device__ int _block_k_515_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_517_ is already defined
#ifndef _block_k_517__func
#define _block_k_517__func
__device__ int _block_k_517_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_519_ is already defined
#ifndef _block_k_519__func
#define _block_k_519__func
__device__ int _block_k_519_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_521_ is already defined
#ifndef _block_k_521__func
#define _block_k_521__func
__device__ int _block_k_521_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_523_ is already defined
#ifndef _block_k_523__func
#define _block_k_523__func
__device__ int _block_k_523_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_525_ is already defined
#ifndef _block_k_525__func
#define _block_k_525__func
__device__ int _block_k_525_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_527_ is already defined
#ifndef _block_k_527__func
#define _block_k_527__func
__device__ int _block_k_527_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_529_ is already defined
#ifndef _block_k_529__func
#define _block_k_529__func
__device__ int _block_k_529_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_531_ is already defined
#ifndef _block_k_531__func
#define _block_k_531__func
__device__ int _block_k_531_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_533_ is already defined
#ifndef _block_k_533__func
#define _block_k_533__func
__device__ int _block_k_533_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_535_ is already defined
#ifndef _block_k_535__func
#define _block_k_535__func
__device__ int _block_k_535_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_537_ is already defined
#ifndef _block_k_537__func
#define _block_k_537__func
__device__ int _block_k_537_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_539_ is already defined
#ifndef _block_k_539__func
#define _block_k_539__func
__device__ int _block_k_539_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_541_ is already defined
#ifndef _block_k_541__func
#define _block_k_541__func
__device__ int _block_k_541_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_543_ is already defined
#ifndef _block_k_543__func
#define _block_k_543__func
__device__ int _block_k_543_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_545_ is already defined
#ifndef _block_k_545__func
#define _block_k_545__func
__device__ int _block_k_545_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_547_ is already defined
#ifndef _block_k_547__func
#define _block_k_547__func
__device__ int _block_k_547_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_549_ is already defined
#ifndef _block_k_549__func
#define _block_k_549__func
__device__ int _block_k_549_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_551_ is already defined
#ifndef _block_k_551__func
#define _block_k_551__func
__device__ int _block_k_551_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_553_ is already defined
#ifndef _block_k_553__func
#define _block_k_553__func
__device__ int _block_k_553_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_555_ is already defined
#ifndef _block_k_555__func
#define _block_k_555__func
__device__ int _block_k_555_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_557_ is already defined
#ifndef _block_k_557__func
#define _block_k_557__func
__device__ int _block_k_557_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_559_ is already defined
#ifndef _block_k_559__func
#define _block_k_559__func
__device__ int _block_k_559_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_561_ is already defined
#ifndef _block_k_561__func
#define _block_k_561__func
__device__ int _block_k_561_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_563_ is already defined
#ifndef _block_k_563__func
#define _block_k_563__func
__device__ int _block_k_563_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_565_ is already defined
#ifndef _block_k_565__func
#define _block_k_565__func
__device__ int _block_k_565_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_567_ is already defined
#ifndef _block_k_567__func
#define _block_k_567__func
__device__ int _block_k_567_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_569_ is already defined
#ifndef _block_k_569__func
#define _block_k_569__func
__device__ int _block_k_569_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_571_ is already defined
#ifndef _block_k_571__func
#define _block_k_571__func
__device__ int _block_k_571_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_573_ is already defined
#ifndef _block_k_573__func
#define _block_k_573__func
__device__ int _block_k_573_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_575_ is already defined
#ifndef _block_k_575__func
#define _block_k_575__func
__device__ int _block_k_575_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_577_ is already defined
#ifndef _block_k_577__func
#define _block_k_577__func
__device__ int _block_k_577_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_579_ is already defined
#ifndef _block_k_579__func
#define _block_k_579__func
__device__ int _block_k_579_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_581_ is already defined
#ifndef _block_k_581__func
#define _block_k_581__func
__device__ int _block_k_581_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_583_ is already defined
#ifndef _block_k_583__func
#define _block_k_583__func
__device__ int _block_k_583_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_585_ is already defined
#ifndef _block_k_585__func
#define _block_k_585__func
__device__ int _block_k_585_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_587_ is already defined
#ifndef _block_k_587__func
#define _block_k_587__func
__device__ int _block_k_587_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_589_ is already defined
#ifndef _block_k_589__func
#define _block_k_589__func
__device__ int _block_k_589_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_591_ is already defined
#ifndef _block_k_591__func
#define _block_k_591__func
__device__ int _block_k_591_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_593_ is already defined
#ifndef _block_k_593__func
#define _block_k_593__func
__device__ int _block_k_593_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_595_ is already defined
#ifndef _block_k_595__func
#define _block_k_595__func
__device__ int _block_k_595_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_597_ is already defined
#ifndef _block_k_597__func
#define _block_k_597__func
__device__ int _block_k_597_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_599_ is already defined
#ifndef _block_k_599__func
#define _block_k_599__func
__device__ int _block_k_599_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_601_ is already defined
#ifndef _block_k_601__func
#define _block_k_601__func
__device__ int _block_k_601_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_603_ is already defined
#ifndef _block_k_603__func
#define _block_k_603__func
__device__ int _block_k_603_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_605_ is already defined
#ifndef _block_k_605__func
#define _block_k_605__func
__device__ int _block_k_605_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_607_ is already defined
#ifndef _block_k_607__func
#define _block_k_607__func
__device__ int _block_k_607_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_609_ is already defined
#ifndef _block_k_609__func
#define _block_k_609__func
__device__ int _block_k_609_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_611_ is already defined
#ifndef _block_k_611__func
#define _block_k_611__func
__device__ int _block_k_611_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_613_ is already defined
#ifndef _block_k_613__func
#define _block_k_613__func
__device__ int _block_k_613_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_615_ is already defined
#ifndef _block_k_615__func
#define _block_k_615__func
__device__ int _block_k_615_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_617_ is already defined
#ifndef _block_k_617__func
#define _block_k_617__func
__device__ int _block_k_617_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_619_ is already defined
#ifndef _block_k_619__func
#define _block_k_619__func
__device__ int _block_k_619_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_621_ is already defined
#ifndef _block_k_621__func
#define _block_k_621__func
__device__ int _block_k_621_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_623_ is already defined
#ifndef _block_k_623__func
#define _block_k_623__func
__device__ int _block_k_623_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_625_ is already defined
#ifndef _block_k_625__func
#define _block_k_625__func
__device__ int _block_k_625_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_627_ is already defined
#ifndef _block_k_627__func
#define _block_k_627__func
__device__ int _block_k_627_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_629_ is already defined
#ifndef _block_k_629__func
#define _block_k_629__func
__device__ int _block_k_629_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_631_ is already defined
#ifndef _block_k_631__func
#define _block_k_631__func
__device__ int _block_k_631_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_633_ is already defined
#ifndef _block_k_633__func
#define _block_k_633__func
__device__ int _block_k_633_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_635_ is already defined
#ifndef _block_k_635__func
#define _block_k_635__func
__device__ int _block_k_635_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_637_ is already defined
#ifndef _block_k_637__func
#define _block_k_637__func
__device__ int _block_k_637_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_639_ is already defined
#ifndef _block_k_639__func
#define _block_k_639__func
__device__ int _block_k_639_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_641_ is already defined
#ifndef _block_k_641__func
#define _block_k_641__func
__device__ int _block_k_641_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_643_ is already defined
#ifndef _block_k_643__func
#define _block_k_643__func
__device__ int _block_k_643_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_645_ is already defined
#ifndef _block_k_645__func
#define _block_k_645__func
__device__ int _block_k_645_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_647_ is already defined
#ifndef _block_k_647__func
#define _block_k_647__func
__device__ int _block_k_647_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_649_ is already defined
#ifndef _block_k_649__func
#define _block_k_649__func
__device__ int _block_k_649_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_651_ is already defined
#ifndef _block_k_651__func
#define _block_k_651__func
__device__ int _block_k_651_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_653_ is already defined
#ifndef _block_k_653__func
#define _block_k_653__func
__device__ int _block_k_653_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_655_ is already defined
#ifndef _block_k_655__func
#define _block_k_655__func
__device__ int _block_k_655_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_657_ is already defined
#ifndef _block_k_657__func
#define _block_k_657__func
__device__ int _block_k_657_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_659_ is already defined
#ifndef _block_k_659__func
#define _block_k_659__func
__device__ int _block_k_659_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_661_ is already defined
#ifndef _block_k_661__func
#define _block_k_661__func
__device__ int _block_k_661_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_663_ is already defined
#ifndef _block_k_663__func
#define _block_k_663__func
__device__ int _block_k_663_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_665_ is already defined
#ifndef _block_k_665__func
#define _block_k_665__func
__device__ int _block_k_665_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_667_ is already defined
#ifndef _block_k_667__func
#define _block_k_667__func
__device__ int _block_k_667_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_669_ is already defined
#ifndef _block_k_669__func
#define _block_k_669__func
__device__ int _block_k_669_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_671_ is already defined
#ifndef _block_k_671__func
#define _block_k_671__func
__device__ int _block_k_671_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_673_ is already defined
#ifndef _block_k_673__func
#define _block_k_673__func
__device__ int _block_k_673_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_675_ is already defined
#ifndef _block_k_675__func
#define _block_k_675__func
__device__ int _block_k_675_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_677_ is already defined
#ifndef _block_k_677__func
#define _block_k_677__func
__device__ int _block_k_677_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_679_ is already defined
#ifndef _block_k_679__func
#define _block_k_679__func
__device__ int _block_k_679_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_681_ is already defined
#ifndef _block_k_681__func
#define _block_k_681__func
__device__ int _block_k_681_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_683_ is already defined
#ifndef _block_k_683__func
#define _block_k_683__func
__device__ int _block_k_683_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_685_ is already defined
#ifndef _block_k_685__func
#define _block_k_685__func
__device__ int _block_k_685_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_687_ is already defined
#ifndef _block_k_687__func
#define _block_k_687__func
__device__ int _block_k_687_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_689_ is already defined
#ifndef _block_k_689__func
#define _block_k_689__func
__device__ int _block_k_689_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_691_ is already defined
#ifndef _block_k_691__func
#define _block_k_691__func
__device__ int _block_k_691_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_693_ is already defined
#ifndef _block_k_693__func
#define _block_k_693__func
__device__ int _block_k_693_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_695_ is already defined
#ifndef _block_k_695__func
#define _block_k_695__func
__device__ int _block_k_695_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_697_ is already defined
#ifndef _block_k_697__func
#define _block_k_697__func
__device__ int _block_k_697_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_699_ is already defined
#ifndef _block_k_699__func
#define _block_k_699__func
__device__ int _block_k_699_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_701_ is already defined
#ifndef _block_k_701__func
#define _block_k_701__func
__device__ int _block_k_701_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_703_ is already defined
#ifndef _block_k_703__func
#define _block_k_703__func
__device__ int _block_k_703_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_705_ is already defined
#ifndef _block_k_705__func
#define _block_k_705__func
__device__ int _block_k_705_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_707_ is already defined
#ifndef _block_k_707__func
#define _block_k_707__func
__device__ int _block_k_707_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_709_ is already defined
#ifndef _block_k_709__func
#define _block_k_709__func
__device__ int _block_k_709_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_711_ is already defined
#ifndef _block_k_711__func
#define _block_k_711__func
__device__ int _block_k_711_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_713_ is already defined
#ifndef _block_k_713__func
#define _block_k_713__func
__device__ int _block_k_713_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_715_ is already defined
#ifndef _block_k_715__func
#define _block_k_715__func
__device__ int _block_k_715_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_717_ is already defined
#ifndef _block_k_717__func
#define _block_k_717__func
__device__ int _block_k_717_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_719_ is already defined
#ifndef _block_k_719__func
#define _block_k_719__func
__device__ int _block_k_719_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_721_ is already defined
#ifndef _block_k_721__func
#define _block_k_721__func
__device__ int _block_k_721_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_723_ is already defined
#ifndef _block_k_723__func
#define _block_k_723__func
__device__ int _block_k_723_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_725_ is already defined
#ifndef _block_k_725__func
#define _block_k_725__func
__device__ int _block_k_725_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_727_ is already defined
#ifndef _block_k_727__func
#define _block_k_727__func
__device__ int _block_k_727_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_729_ is already defined
#ifndef _block_k_729__func
#define _block_k_729__func
__device__ int _block_k_729_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_731_ is already defined
#ifndef _block_k_731__func
#define _block_k_731__func
__device__ int _block_k_731_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_733_ is already defined
#ifndef _block_k_733__func
#define _block_k_733__func
__device__ int _block_k_733_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_735_ is already defined
#ifndef _block_k_735__func
#define _block_k_735__func
__device__ int _block_k_735_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_737_ is already defined
#ifndef _block_k_737__func
#define _block_k_737__func
__device__ int _block_k_737_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_739_ is already defined
#ifndef _block_k_739__func
#define _block_k_739__func
__device__ int _block_k_739_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_741_ is already defined
#ifndef _block_k_741__func
#define _block_k_741__func
__device__ int _block_k_741_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_743_ is already defined
#ifndef _block_k_743__func
#define _block_k_743__func
__device__ int _block_k_743_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_745_ is already defined
#ifndef _block_k_745__func
#define _block_k_745__func
__device__ int _block_k_745_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_747_ is already defined
#ifndef _block_k_747__func
#define _block_k_747__func
__device__ int _block_k_747_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_749_ is already defined
#ifndef _block_k_749__func
#define _block_k_749__func
__device__ int _block_k_749_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_751_ is already defined
#ifndef _block_k_751__func
#define _block_k_751__func
__device__ int _block_k_751_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_753_ is already defined
#ifndef _block_k_753__func
#define _block_k_753__func
__device__ int _block_k_753_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_755_ is already defined
#ifndef _block_k_755__func
#define _block_k_755__func
__device__ int _block_k_755_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_757_ is already defined
#ifndef _block_k_757__func
#define _block_k_757__func
__device__ int _block_k_757_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_759_ is already defined
#ifndef _block_k_759__func
#define _block_k_759__func
__device__ int _block_k_759_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_761_ is already defined
#ifndef _block_k_761__func
#define _block_k_761__func
__device__ int _block_k_761_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_763_ is already defined
#ifndef _block_k_763__func
#define _block_k_763__func
__device__ int _block_k_763_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_765_ is already defined
#ifndef _block_k_765__func
#define _block_k_765__func
__device__ int _block_k_765_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_767_ is already defined
#ifndef _block_k_767__func
#define _block_k_767__func
__device__ int _block_k_767_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_769_ is already defined
#ifndef _block_k_769__func
#define _block_k_769__func
__device__ int _block_k_769_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_771_ is already defined
#ifndef _block_k_771__func
#define _block_k_771__func
__device__ int _block_k_771_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_773_ is already defined
#ifndef _block_k_773__func
#define _block_k_773__func
__device__ int _block_k_773_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_775_ is already defined
#ifndef _block_k_775__func
#define _block_k_775__func
__device__ int _block_k_775_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_777_ is already defined
#ifndef _block_k_777__func
#define _block_k_777__func
__device__ int _block_k_777_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_779_ is already defined
#ifndef _block_k_779__func
#define _block_k_779__func
__device__ int _block_k_779_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_781_ is already defined
#ifndef _block_k_781__func
#define _block_k_781__func
__device__ int _block_k_781_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_783_ is already defined
#ifndef _block_k_783__func
#define _block_k_783__func
__device__ int _block_k_783_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_785_ is already defined
#ifndef _block_k_785__func
#define _block_k_785__func
__device__ int _block_k_785_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_787_ is already defined
#ifndef _block_k_787__func
#define _block_k_787__func
__device__ int _block_k_787_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_789_ is already defined
#ifndef _block_k_789__func
#define _block_k_789__func
__device__ int _block_k_789_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_791_ is already defined
#ifndef _block_k_791__func
#define _block_k_791__func
__device__ int _block_k_791_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_793_ is already defined
#ifndef _block_k_793__func
#define _block_k_793__func
__device__ int _block_k_793_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_795_ is already defined
#ifndef _block_k_795__func
#define _block_k_795__func
__device__ int _block_k_795_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_797_ is already defined
#ifndef _block_k_797__func
#define _block_k_797__func
__device__ int _block_k_797_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_799_ is already defined
#ifndef _block_k_799__func
#define _block_k_799__func
__device__ int _block_k_799_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_801_ is already defined
#ifndef _block_k_801__func
#define _block_k_801__func
__device__ int _block_k_801_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_803_ is already defined
#ifndef _block_k_803__func
#define _block_k_803__func
__device__ int _block_k_803_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_805_ is already defined
#ifndef _block_k_805__func
#define _block_k_805__func
__device__ int _block_k_805_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_807_ is already defined
#ifndef _block_k_807__func
#define _block_k_807__func
__device__ int _block_k_807_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_809_ is already defined
#ifndef _block_k_809__func
#define _block_k_809__func
__device__ int _block_k_809_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_811_ is already defined
#ifndef _block_k_811__func
#define _block_k_811__func
__device__ int _block_k_811_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_813_ is already defined
#ifndef _block_k_813__func
#define _block_k_813__func
__device__ int _block_k_813_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_815_ is already defined
#ifndef _block_k_815__func
#define _block_k_815__func
__device__ int _block_k_815_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_817_ is already defined
#ifndef _block_k_817__func
#define _block_k_817__func
__device__ int _block_k_817_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_819_ is already defined
#ifndef _block_k_819__func
#define _block_k_819__func
__device__ int _block_k_819_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_821_ is already defined
#ifndef _block_k_821__func
#define _block_k_821__func
__device__ int _block_k_821_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_823_ is already defined
#ifndef _block_k_823__func
#define _block_k_823__func
__device__ int _block_k_823_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_825_ is already defined
#ifndef _block_k_825__func
#define _block_k_825__func
__device__ int _block_k_825_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_827_ is already defined
#ifndef _block_k_827__func
#define _block_k_827__func
__device__ int _block_k_827_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_829_ is already defined
#ifndef _block_k_829__func
#define _block_k_829__func
__device__ int _block_k_829_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_831_ is already defined
#ifndef _block_k_831__func
#define _block_k_831__func
__device__ int _block_k_831_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_833_ is already defined
#ifndef _block_k_833__func
#define _block_k_833__func
__device__ int _block_k_833_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_835_ is already defined
#ifndef _block_k_835__func
#define _block_k_835__func
__device__ int _block_k_835_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_837_ is already defined
#ifndef _block_k_837__func
#define _block_k_837__func
__device__ int _block_k_837_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_839_ is already defined
#ifndef _block_k_839__func
#define _block_k_839__func
__device__ int _block_k_839_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_841_ is already defined
#ifndef _block_k_841__func
#define _block_k_841__func
__device__ int _block_k_841_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_843_ is already defined
#ifndef _block_k_843__func
#define _block_k_843__func
__device__ int _block_k_843_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_845_ is already defined
#ifndef _block_k_845__func
#define _block_k_845__func
__device__ int _block_k_845_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_847_ is already defined
#ifndef _block_k_847__func
#define _block_k_847__func
__device__ int _block_k_847_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_849_ is already defined
#ifndef _block_k_849__func
#define _block_k_849__func
__device__ int _block_k_849_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_851_ is already defined
#ifndef _block_k_851__func
#define _block_k_851__func
__device__ int _block_k_851_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_853_ is already defined
#ifndef _block_k_853__func
#define _block_k_853__func
__device__ int _block_k_853_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_855_ is already defined
#ifndef _block_k_855__func
#define _block_k_855__func
__device__ int _block_k_855_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_857_ is already defined
#ifndef _block_k_857__func
#define _block_k_857__func
__device__ int _block_k_857_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_859_ is already defined
#ifndef _block_k_859__func
#define _block_k_859__func
__device__ int _block_k_859_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_861_ is already defined
#ifndef _block_k_861__func
#define _block_k_861__func
__device__ int _block_k_861_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_863_ is already defined
#ifndef _block_k_863__func
#define _block_k_863__func
__device__ int _block_k_863_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_865_ is already defined
#ifndef _block_k_865__func
#define _block_k_865__func
__device__ int _block_k_865_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_867_ is already defined
#ifndef _block_k_867__func
#define _block_k_867__func
__device__ int _block_k_867_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_869_ is already defined
#ifndef _block_k_869__func
#define _block_k_869__func
__device__ int _block_k_869_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_871_ is already defined
#ifndef _block_k_871__func
#define _block_k_871__func
__device__ int _block_k_871_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_873_ is already defined
#ifndef _block_k_873__func
#define _block_k_873__func
__device__ int _block_k_873_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_875_ is already defined
#ifndef _block_k_875__func
#define _block_k_875__func
__device__ int _block_k_875_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_877_ is already defined
#ifndef _block_k_877__func
#define _block_k_877__func
__device__ int _block_k_877_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_879_ is already defined
#ifndef _block_k_879__func
#define _block_k_879__func
__device__ int _block_k_879_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_881_ is already defined
#ifndef _block_k_881__func
#define _block_k_881__func
__device__ int _block_k_881_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_883_ is already defined
#ifndef _block_k_883__func
#define _block_k_883__func
__device__ int _block_k_883_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_885_ is already defined
#ifndef _block_k_885__func
#define _block_k_885__func
__device__ int _block_k_885_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_887_ is already defined
#ifndef _block_k_887__func
#define _block_k_887__func
__device__ int _block_k_887_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_889_ is already defined
#ifndef _block_k_889__func
#define _block_k_889__func
__device__ int _block_k_889_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_891_ is already defined
#ifndef _block_k_891__func
#define _block_k_891__func
__device__ int _block_k_891_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_893_ is already defined
#ifndef _block_k_893__func
#define _block_k_893__func
__device__ int _block_k_893_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_895_ is already defined
#ifndef _block_k_895__func
#define _block_k_895__func
__device__ int _block_k_895_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_897_ is already defined
#ifndef _block_k_897__func
#define _block_k_897__func
__device__ int _block_k_897_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_899_ is already defined
#ifndef _block_k_899__func
#define _block_k_899__func
__device__ int _block_k_899_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_901_ is already defined
#ifndef _block_k_901__func
#define _block_k_901__func
__device__ int _block_k_901_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_903_ is already defined
#ifndef _block_k_903__func
#define _block_k_903__func
__device__ int _block_k_903_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_905_ is already defined
#ifndef _block_k_905__func
#define _block_k_905__func
__device__ int _block_k_905_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_907_ is already defined
#ifndef _block_k_907__func
#define _block_k_907__func
__device__ int _block_k_907_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_909_ is already defined
#ifndef _block_k_909__func
#define _block_k_909__func
__device__ int _block_k_909_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_911_ is already defined
#ifndef _block_k_911__func
#define _block_k_911__func
__device__ int _block_k_911_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_913_ is already defined
#ifndef _block_k_913__func
#define _block_k_913__func
__device__ int _block_k_913_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_915_ is already defined
#ifndef _block_k_915__func
#define _block_k_915__func
__device__ int _block_k_915_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_917_ is already defined
#ifndef _block_k_917__func
#define _block_k_917__func
__device__ int _block_k_917_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_919_ is already defined
#ifndef _block_k_919__func
#define _block_k_919__func
__device__ int _block_k_919_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_921_ is already defined
#ifndef _block_k_921__func
#define _block_k_921__func
__device__ int _block_k_921_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_923_ is already defined
#ifndef _block_k_923__func
#define _block_k_923__func
__device__ int _block_k_923_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_925_ is already defined
#ifndef _block_k_925__func
#define _block_k_925__func
__device__ int _block_k_925_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_927_ is already defined
#ifndef _block_k_927__func
#define _block_k_927__func
__device__ int _block_k_927_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_929_ is already defined
#ifndef _block_k_929__func
#define _block_k_929__func
__device__ int _block_k_929_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_931_ is already defined
#ifndef _block_k_931__func
#define _block_k_931__func
__device__ int _block_k_931_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_933_ is already defined
#ifndef _block_k_933__func
#define _block_k_933__func
__device__ int _block_k_933_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_935_ is already defined
#ifndef _block_k_935__func
#define _block_k_935__func
__device__ int _block_k_935_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_937_ is already defined
#ifndef _block_k_937__func
#define _block_k_937__func
__device__ int _block_k_937_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_939_ is already defined
#ifndef _block_k_939__func
#define _block_k_939__func
__device__ int _block_k_939_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_941_ is already defined
#ifndef _block_k_941__func
#define _block_k_941__func
__device__ int _block_k_941_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_943_ is already defined
#ifndef _block_k_943__func
#define _block_k_943__func
__device__ int _block_k_943_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_945_ is already defined
#ifndef _block_k_945__func
#define _block_k_945__func
__device__ int _block_k_945_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_947_ is already defined
#ifndef _block_k_947__func
#define _block_k_947__func
__device__ int _block_k_947_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_949_ is already defined
#ifndef _block_k_949__func
#define _block_k_949__func
__device__ int _block_k_949_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_951_ is already defined
#ifndef _block_k_951__func
#define _block_k_951__func
__device__ int _block_k_951_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_953_ is already defined
#ifndef _block_k_953__func
#define _block_k_953__func
__device__ int _block_k_953_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_955_ is already defined
#ifndef _block_k_955__func
#define _block_k_955__func
__device__ int _block_k_955_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_957_ is already defined
#ifndef _block_k_957__func
#define _block_k_957__func
__device__ int _block_k_957_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_959_ is already defined
#ifndef _block_k_959__func
#define _block_k_959__func
__device__ int _block_k_959_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_961_ is already defined
#ifndef _block_k_961__func
#define _block_k_961__func
__device__ int _block_k_961_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_963_ is already defined
#ifndef _block_k_963__func
#define _block_k_963__func
__device__ int _block_k_963_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_965_ is already defined
#ifndef _block_k_965__func
#define _block_k_965__func
__device__ int _block_k_965_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_967_ is already defined
#ifndef _block_k_967__func
#define _block_k_967__func
__device__ int _block_k_967_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_969_ is already defined
#ifndef _block_k_969__func
#define _block_k_969__func
__device__ int _block_k_969_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_971_ is already defined
#ifndef _block_k_971__func
#define _block_k_971__func
__device__ int _block_k_971_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_973_ is already defined
#ifndef _block_k_973__func
#define _block_k_973__func
__device__ int _block_k_973_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_975_ is already defined
#ifndef _block_k_975__func
#define _block_k_975__func
__device__ int _block_k_975_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_977_ is already defined
#ifndef _block_k_977__func
#define _block_k_977__func
__device__ int _block_k_977_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_979_ is already defined
#ifndef _block_k_979__func
#define _block_k_979__func
__device__ int _block_k_979_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_981_ is already defined
#ifndef _block_k_981__func
#define _block_k_981__func
__device__ int _block_k_981_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_983_ is already defined
#ifndef _block_k_983__func
#define _block_k_983__func
__device__ int _block_k_983_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_0)) % 11799);
    }
}

#endif



// TODO: There should be a better to check if _block_k_985_ is already defined
#ifndef _block_k_985__func
#define _block_k_985__func
__device__ int _block_k_985_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_987_ is already defined
#ifndef _block_k_987__func
#define _block_k_987__func
__device__ int _block_k_987_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_989_ is already defined
#ifndef _block_k_989__func
#define _block_k_989__func
__device__ int _block_k_989_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_991_ is already defined
#ifndef _block_k_991__func
#define _block_k_991__func
__device__ int _block_k_991_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_993_ is already defined
#ifndef _block_k_993__func
#define _block_k_993__func
__device__ int _block_k_993_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_3)) % 77689);
    }
}

#endif



// TODO: There should be a better to check if _block_k_995_ is already defined
#ifndef _block_k_995__func
#define _block_k_995__func
__device__ int _block_k_995_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif



// TODO: There should be a better to check if _block_k_997_ is already defined
#ifndef _block_k_997__func
#define _block_k_997__func
__device__ int _block_k_997_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 1337);
    }
}

#endif



// TODO: There should be a better to check if _block_k_999_ is already defined
#ifndef _block_k_999__func
#define _block_k_999__func
__device__ int _block_k_999_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 8888888);
    }
}

#endif



// TODO: There should be a better to check if _block_k_1001_ is already defined
#ifndef _block_k_1001__func
#define _block_k_1001__func
__device__ int _block_k_1001_(environment_t *_env_, int i, indexed_struct_4_lt_int_int_int_int_gt_t indices)
{
    
    
    
    {
        return (((i + indices.field_2)) % 6678);
    }
}

#endif


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_1001_(_env_, _block_k_999_(_env_, _block_k_997_(_env_, _block_k_995_(_env_, _block_k_993_(_env_, _block_k_991_(_env_, _block_k_989_(_env_, _block_k_987_(_env_, _block_k_985_(_env_, _block_k_983_(_env_, _block_k_981_(_env_, _block_k_979_(_env_, _block_k_977_(_env_, _block_k_975_(_env_, _block_k_973_(_env_, _block_k_971_(_env_, _block_k_969_(_env_, _block_k_967_(_env_, _block_k_965_(_env_, _block_k_963_(_env_, _block_k_961_(_env_, _block_k_959_(_env_, _block_k_957_(_env_, _block_k_955_(_env_, _block_k_953_(_env_, _block_k_951_(_env_, _block_k_949_(_env_, _block_k_947_(_env_, _block_k_945_(_env_, _block_k_943_(_env_, _block_k_941_(_env_, _block_k_939_(_env_, _block_k_937_(_env_, _block_k_935_(_env_, _block_k_933_(_env_, _block_k_931_(_env_, _block_k_929_(_env_, _block_k_927_(_env_, _block_k_925_(_env_, _block_k_923_(_env_, _block_k_921_(_env_, _block_k_919_(_env_, _block_k_917_(_env_, _block_k_915_(_env_, _block_k_913_(_env_, _block_k_911_(_env_, _block_k_909_(_env_, _block_k_907_(_env_, _block_k_905_(_env_, _block_k_903_(_env_, _block_k_901_(_env_, _block_k_899_(_env_, _block_k_897_(_env_, _block_k_895_(_env_, _block_k_893_(_env_, _block_k_891_(_env_, _block_k_889_(_env_, _block_k_887_(_env_, _block_k_885_(_env_, _block_k_883_(_env_, _block_k_881_(_env_, _block_k_879_(_env_, _block_k_877_(_env_, _block_k_875_(_env_, _block_k_873_(_env_, _block_k_871_(_env_, _block_k_869_(_env_, _block_k_867_(_env_, _block_k_865_(_env_, _block_k_863_(_env_, _block_k_861_(_env_, _block_k_859_(_env_, _block_k_857_(_env_, _block_k_855_(_env_, _block_k_853_(_env_, _block_k_851_(_env_, _block_k_849_(_env_, _block_k_847_(_env_, _block_k_845_(_env_, _block_k_843_(_env_, _block_k_841_(_env_, _block_k_839_(_env_, _block_k_837_(_env_, _block_k_835_(_env_, _block_k_833_(_env_, _block_k_831_(_env_, _block_k_829_(_env_, _block_k_827_(_env_, _block_k_825_(_env_, _block_k_823_(_env_, _block_k_821_(_env_, _block_k_819_(_env_, _block_k_817_(_env_, _block_k_815_(_env_, _block_k_813_(_env_, _block_k_811_(_env_, _block_k_809_(_env_, _block_k_807_(_env_, _block_k_805_(_env_, _block_k_803_(_env_, _block_k_801_(_env_, _block_k_799_(_env_, _block_k_797_(_env_, _block_k_795_(_env_, _block_k_793_(_env_, _block_k_791_(_env_, _block_k_789_(_env_, _block_k_787_(_env_, _block_k_785_(_env_, _block_k_783_(_env_, _block_k_781_(_env_, _block_k_779_(_env_, _block_k_777_(_env_, _block_k_775_(_env_, _block_k_773_(_env_, _block_k_771_(_env_, _block_k_769_(_env_, _block_k_767_(_env_, _block_k_765_(_env_, _block_k_763_(_env_, _block_k_761_(_env_, _block_k_759_(_env_, _block_k_757_(_env_, _block_k_755_(_env_, _block_k_753_(_env_, _block_k_751_(_env_, _block_k_749_(_env_, _block_k_747_(_env_, _block_k_745_(_env_, _block_k_743_(_env_, _block_k_741_(_env_, _block_k_739_(_env_, _block_k_737_(_env_, _block_k_735_(_env_, _block_k_733_(_env_, _block_k_731_(_env_, _block_k_729_(_env_, _block_k_727_(_env_, _block_k_725_(_env_, _block_k_723_(_env_, _block_k_721_(_env_, _block_k_719_(_env_, _block_k_717_(_env_, _block_k_715_(_env_, _block_k_713_(_env_, _block_k_711_(_env_, _block_k_709_(_env_, _block_k_707_(_env_, _block_k_705_(_env_, _block_k_703_(_env_, _block_k_701_(_env_, _block_k_699_(_env_, _block_k_697_(_env_, _block_k_695_(_env_, _block_k_693_(_env_, _block_k_691_(_env_, _block_k_689_(_env_, _block_k_687_(_env_, _block_k_685_(_env_, _block_k_683_(_env_, _block_k_681_(_env_, _block_k_679_(_env_, _block_k_677_(_env_, _block_k_675_(_env_, _block_k_673_(_env_, _block_k_671_(_env_, _block_k_669_(_env_, _block_k_667_(_env_, _block_k_665_(_env_, _block_k_663_(_env_, _block_k_661_(_env_, _block_k_659_(_env_, _block_k_657_(_env_, _block_k_655_(_env_, _block_k_653_(_env_, _block_k_651_(_env_, _block_k_649_(_env_, _block_k_647_(_env_, _block_k_645_(_env_, _block_k_643_(_env_, _block_k_641_(_env_, _block_k_639_(_env_, _block_k_637_(_env_, _block_k_635_(_env_, _block_k_633_(_env_, _block_k_631_(_env_, _block_k_629_(_env_, _block_k_627_(_env_, _block_k_625_(_env_, _block_k_623_(_env_, _block_k_621_(_env_, _block_k_619_(_env_, _block_k_617_(_env_, _block_k_615_(_env_, _block_k_613_(_env_, _block_k_611_(_env_, _block_k_609_(_env_, _block_k_607_(_env_, _block_k_605_(_env_, _block_k_603_(_env_, _block_k_601_(_env_, _block_k_599_(_env_, _block_k_597_(_env_, _block_k_595_(_env_, _block_k_593_(_env_, _block_k_591_(_env_, _block_k_589_(_env_, _block_k_587_(_env_, _block_k_585_(_env_, _block_k_583_(_env_, _block_k_581_(_env_, _block_k_579_(_env_, _block_k_577_(_env_, _block_k_575_(_env_, _block_k_573_(_env_, _block_k_571_(_env_, _block_k_569_(_env_, _block_k_567_(_env_, _block_k_565_(_env_, _block_k_563_(_env_, _block_k_561_(_env_, _block_k_559_(_env_, _block_k_557_(_env_, _block_k_555_(_env_, _block_k_553_(_env_, _block_k_551_(_env_, _block_k_549_(_env_, _block_k_547_(_env_, _block_k_545_(_env_, _block_k_543_(_env_, _block_k_541_(_env_, _block_k_539_(_env_, _block_k_537_(_env_, _block_k_535_(_env_, _block_k_533_(_env_, _block_k_531_(_env_, _block_k_529_(_env_, _block_k_527_(_env_, _block_k_525_(_env_, _block_k_523_(_env_, _block_k_521_(_env_, _block_k_519_(_env_, _block_k_517_(_env_, _block_k_515_(_env_, _block_k_513_(_env_, _block_k_511_(_env_, _block_k_509_(_env_, _block_k_507_(_env_, _block_k_505_(_env_, _block_k_503_(_env_, _block_k_501_(_env_, _block_k_499_(_env_, _block_k_497_(_env_, _block_k_495_(_env_, _block_k_493_(_env_, _block_k_491_(_env_, _block_k_489_(_env_, _block_k_487_(_env_, _block_k_485_(_env_, _block_k_483_(_env_, _block_k_481_(_env_, _block_k_479_(_env_, _block_k_477_(_env_, _block_k_475_(_env_, _block_k_473_(_env_, _block_k_471_(_env_, _block_k_469_(_env_, _block_k_467_(_env_, _block_k_465_(_env_, _block_k_463_(_env_, _block_k_461_(_env_, _block_k_459_(_env_, _block_k_457_(_env_, _block_k_455_(_env_, _block_k_453_(_env_, _block_k_451_(_env_, _block_k_449_(_env_, _block_k_447_(_env_, _block_k_445_(_env_, _block_k_443_(_env_, _block_k_441_(_env_, _block_k_439_(_env_, _block_k_437_(_env_, _block_k_435_(_env_, _block_k_433_(_env_, _block_k_431_(_env_, _block_k_429_(_env_, _block_k_427_(_env_, _block_k_425_(_env_, _block_k_423_(_env_, _block_k_421_(_env_, _block_k_419_(_env_, _block_k_417_(_env_, _block_k_415_(_env_, _block_k_413_(_env_, _block_k_411_(_env_, _block_k_409_(_env_, _block_k_407_(_env_, _block_k_405_(_env_, _block_k_403_(_env_, _block_k_401_(_env_, _block_k_399_(_env_, _block_k_397_(_env_, _block_k_395_(_env_, _block_k_393_(_env_, _block_k_391_(_env_, _block_k_389_(_env_, _block_k_387_(_env_, _block_k_385_(_env_, _block_k_383_(_env_, _block_k_381_(_env_, _block_k_379_(_env_, _block_k_377_(_env_, _block_k_375_(_env_, _block_k_373_(_env_, _block_k_371_(_env_, _block_k_369_(_env_, _block_k_367_(_env_, _block_k_365_(_env_, _block_k_363_(_env_, _block_k_361_(_env_, _block_k_359_(_env_, _block_k_357_(_env_, _block_k_355_(_env_, _block_k_353_(_env_, _block_k_351_(_env_, _block_k_349_(_env_, _block_k_347_(_env_, _block_k_345_(_env_, _block_k_343_(_env_, _block_k_341_(_env_, _block_k_339_(_env_, _block_k_337_(_env_, _block_k_335_(_env_, _block_k_333_(_env_, _block_k_331_(_env_, _block_k_329_(_env_, _block_k_327_(_env_, _block_k_325_(_env_, _block_k_323_(_env_, _block_k_321_(_env_, _block_k_319_(_env_, _block_k_317_(_env_, _block_k_315_(_env_, _block_k_313_(_env_, _block_k_311_(_env_, _block_k_309_(_env_, _block_k_307_(_env_, _block_k_305_(_env_, _block_k_303_(_env_, _block_k_301_(_env_, _block_k_299_(_env_, _block_k_297_(_env_, _block_k_295_(_env_, _block_k_293_(_env_, _block_k_291_(_env_, _block_k_289_(_env_, _block_k_287_(_env_, _block_k_285_(_env_, _block_k_283_(_env_, _block_k_281_(_env_, _block_k_279_(_env_, _block_k_277_(_env_, _block_k_275_(_env_, _block_k_273_(_env_, _block_k_271_(_env_, _block_k_269_(_env_, _block_k_267_(_env_, _block_k_265_(_env_, _block_k_263_(_env_, _block_k_261_(_env_, _block_k_259_(_env_, _block_k_257_(_env_, _block_k_255_(_env_, _block_k_253_(_env_, _block_k_251_(_env_, _block_k_249_(_env_, _block_k_247_(_env_, _block_k_245_(_env_, _block_k_243_(_env_, _block_k_241_(_env_, _block_k_239_(_env_, _block_k_237_(_env_, _block_k_235_(_env_, _block_k_233_(_env_, _block_k_231_(_env_, _block_k_229_(_env_, _block_k_227_(_env_, _block_k_225_(_env_, _block_k_223_(_env_, _block_k_221_(_env_, _block_k_219_(_env_, _block_k_217_(_env_, _block_k_215_(_env_, _block_k_213_(_env_, _block_k_211_(_env_, _block_k_209_(_env_, _block_k_207_(_env_, _block_k_205_(_env_, _block_k_203_(_env_, _block_k_201_(_env_, _block_k_199_(_env_, _block_k_197_(_env_, _block_k_195_(_env_, _block_k_193_(_env_, _block_k_191_(_env_, _block_k_189_(_env_, _block_k_187_(_env_, _block_k_185_(_env_, _block_k_183_(_env_, _block_k_181_(_env_, _block_k_179_(_env_, _block_k_177_(_env_, _block_k_175_(_env_, _block_k_173_(_env_, _block_k_171_(_env_, _block_k_169_(_env_, _block_k_167_(_env_, _block_k_165_(_env_, _block_k_163_(_env_, _block_k_161_(_env_, _block_k_159_(_env_, _block_k_157_(_env_, _block_k_155_(_env_, _block_k_153_(_env_, _block_k_151_(_env_, _block_k_149_(_env_, _block_k_147_(_env_, _block_k_145_(_env_, _block_k_143_(_env_, _block_k_141_(_env_, _block_k_139_(_env_, _block_k_137_(_env_, _block_k_135_(_env_, _block_k_133_(_env_, _block_k_131_(_env_, _block_k_129_(_env_, _block_k_127_(_env_, _block_k_125_(_env_, _block_k_123_(_env_, _block_k_121_(_env_, _block_k_119_(_env_, _block_k_117_(_env_, _block_k_115_(_env_, _block_k_113_(_env_, _block_k_111_(_env_, _block_k_109_(_env_, _block_k_107_(_env_, _block_k_105_(_env_, _block_k_103_(_env_, _block_k_101_(_env_, _block_k_99_(_env_, _block_k_97_(_env_, _block_k_95_(_env_, _block_k_93_(_env_, _block_k_91_(_env_, _block_k_89_(_env_, _block_k_87_(_env_, _block_k_85_(_env_, _block_k_83_(_env_, _block_k_81_(_env_, _block_k_79_(_env_, _block_k_77_(_env_, _block_k_75_(_env_, _block_k_73_(_env_, _block_k_71_(_env_, _block_k_69_(_env_, _block_k_67_(_env_, _block_k_65_(_env_, _block_k_63_(_env_, _block_k_61_(_env_, _block_k_59_(_env_, _block_k_57_(_env_, _block_k_55_(_env_, _block_k_53_(_env_, _block_k_51_(_env_, _block_k_49_(_env_, _block_k_47_(_env_, _block_k_45_(_env_, _block_k_43_(_env_, _block_k_41_(_env_, _block_k_39_(_env_, _block_k_37_(_env_, _block_k_35_(_env_, _block_k_33_(_env_, _block_k_31_(_env_, _block_k_29_(_env_, _block_k_27_(_env_, _block_k_25_(_env_, _block_k_23_(_env_, _block_k_21_(_env_, _block_k_19_(_env_, _block_k_17_(_env_, _block_k_15_(_env_, _block_k_13_(_env_, _block_k_11_(_env_, _block_k_9_(_env_, _block_k_7_(_env_, _block_k_5_(_env_, _block_k_3_(_env_, _block_k_2_(_env_, ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12})), ((indexed_struct_4_lt_int_int_int_int_gt_t) {_tid_ / 3000000, (_tid_ / 6000) % 500, (_tid_ / 12) % 500, (_tid_ / 1) % 12}));
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
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (sizeof(int) * 60000000)));
    program_result->device_allocations->push_back(_kernel_result_2);
    timeReportMeasure(program_result, allocate_memory);
    timeStartMeasure();
    kernel_1<<<58594, 1024>>>(dev_env, 60000000, _kernel_result_2);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);

    /* Copy over result to the host */
    program_result->result = ({
    variable_size_array_t device_array = variable_size_array_t((void *) _kernel_result_2, 60000000);
    int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
    timeReportMeasure(program_result, transfer_memory);

    variable_size_array_t((void *) tmp_result, device_array.size);
});

    /* Free device memory */
        timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_2));
    program_result->device_allocations->erase(
        std::remove(
            program_result->device_allocations->begin(),
            program_result->device_allocations->end(),
            _kernel_result_2),
        program_result->device_allocations->end());
    timeReportMeasure(program_result, free_memory);


    delete program_result->device_allocations;
    
    return program_result;
}
