/* ----- BEGIN Structs ----- */
template <typename T>
struct array_command_t {
    T *result;
};

/* ----- BEGIN Union Type ----- */
typedef union union_type_value {
    obj_id_t object_id;
    int int_;
    float float_;
    bool bool_;
    array_command_t<void> *array_command;
} union_v_t;

typedef struct union_type_struct
{
    class_id_t class_id;
    union_v_t value;
} union_t;
/* ----- END Union Type ----- */

template <typename T>
struct fixed_size_array_t {
    T *content;
    int size;

    fixed_size_array_t(T *content_, int size_) : content(content_), size(size_) { }; 
};

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

