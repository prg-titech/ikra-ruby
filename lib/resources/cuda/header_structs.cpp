/* ----- BEGIN Structs ----- */
template <typename T>
struct array_command_t {
    T *result;
};

template <typename T>
struct fixed_size_array_t {
    T *content;
    int size;

    fixed_size_array_t(T *content_, int size_) : content(content_), size(size_) { }; 

    static const fixed_size_array_t<T> error_return_value;
};

// error_return_value is used in case a host section terminates abnormally
template <typename T>
const fixed_size_array_t<T> fixed_size_array_t<T>::error_return_value = 
    fixed_size_array_t<T>(NULL, 0);

/* ----- BEGIN Union Type ----- */
typedef union union_type_value {
    obj_id_t object_id;
    int int_;
    float float_;
    bool bool_;
    array_command_t<void> *array_command;
    fixed_size_array_t<void> fixed_size_array;

    __host__ __device__ union_type_value(int value) : int_(value) { };
    __host__ __device__ union_type_value(float value) : float_(value) { };
    __host__ __device__ union_type_value(bool value) : bool_(value) { };
    __host__ __device__ union_type_value(array_command_t<void> *value) : array_command(value) { };
    __host__ __device__ union_type_value(fixed_size_array_t<void> value) : fixed_size_array(value) { };

    __host__ __device__ static union_type_value from_object_id(obj_id_t value) {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_int(int value) {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_float(float value) {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_bool(bool value) {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_array_command_t(array_command_t<void> *value) {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_fixed_size_array_t(fixed_size_array_t<void> value) {
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
    /*{result_type}*/ result;
    int last_error;

    uint64_t time_setup_cuda;
    uint64_t time_prepare_env;
    uint64_t time_kernel;
    uint64_t time_free_memory;

    // Memory management
    vector<void*> *device_allocations;
} result_t;
/* ----- END Structs ----- */

