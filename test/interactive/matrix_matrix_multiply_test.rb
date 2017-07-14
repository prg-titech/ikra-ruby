def generate_matrix(size, seed)
    return Array.new(size * size) do |i|
        (i * 17 + 6.0) % (seed + 27)
    end
end

def generate_identity_matrix(size)
    return Array.new(size * size) do |i|
        if i / size == i % size
            1.0
        else
            0.0
        end
    end
end

# Transposed `matrix` with dimensions size x size
def transposed(matrix, size)
    # TODO (Step 2): Implement on CPU
    return [1.0, 2.0, 3.0, 4.0]
end

def assert_same_values(c1, c2)
    if c1.size != c2.size
        raise "Matrices do not have the same size"
    end

    for i in 0...(c1.size)
        if (c1[i] - c2[i]).abs > 0.01
            raise "Matrices are not equal"
        end
    end

    puts "Matrix equality check passed"
end

# CPU-based implementaton
def matrix_multiplication_cpu(a, b, size)
    return Array.new(size * size) do |index|
        x = index % size
        y = index / size

        # TODO (Step 1): Implement

        123.4
    end
end

# GPU-based implementation
def matrix_multiplication_gpu(a, b, size)
    # Same as: (0...(size*size)).to_a.to_pa.map do |index|
    return PArray.new(size * size) do |index|
        x = index % size
        y = index / size

        # TODO (Step 3): Implement
        # Try with arguments GPU 150
        # If it works: try with arguments GPU 3000 and compare with CPU

        123.4
    end
end

# GPU-based implementation with blocking
# Step 6
def matrix_multiplication_gpu_blocked(a, b, size)
    block_size_x = 15
    block_size_y = 15

    # Get it running with GPU_BLOCK 150 first
    # Then measure runtime with GPU_BLOCK 3000
    
    # Create blocked indices
    # For example (2x2): [0, 1, 4, 5, 2, 3, 6, 7 , ...]
    indices = Array.new(size * size)

    # TODO (Step 6.1): Generate block indices

    unordered_result = indices.to_pa.map do |index|
        x = index % size
        y = index / size

        # TODO (Step 6.2): Implement (same as Step 3)

        123.4
    end

    correct_order_result = Array.new(size*size)

    # TODO (Step 6.3): Restore correct order

    return correct_order_result
end

# GPU-based implementation where b is transposed
def matrix_multiplication_gpu_transposed_b(a, b, size)
    # Matrix B is transposed

    return PArray.new(size * size) do |index|
        x = index % size
        y = index / size

        # TODO(5): Implement
        # Try with GPU_T_B 150
        # If correct, run with GPU_T_B 3000 and compare runtime with GPU 3000

        123.4
    end
end

# GPU-based implementation where a is transposed
def matrix_multiplication_gpu_transposed_a(a, b, size)
    # Matrix A is transposed

    return PArray.new(size * size) do |index|
        x = index % size
        y = index / size

        # TODO(4): Implement
        # Try with GPU_T_A 150
        # If correct, run with GPU_T_A 3000 and compare runtime with GPU 3000

        123.4
    end
end

mode = ARGV[0]
size = ARGV[1].to_i

if size % 150 != 0 or size == 0
    # This limitation is due to current implementation issues in Ikra
    raise "Size must be a multiple of 150"
end

puts "Generating input matrices..."
a = generate_matrix(size, 0)
b = generate_matrix(size, 2)


if mode == "CHECK"
    b = generate_identity_matrix(size)

    puts "CPU Computation, wait..."
    c = matrix_multiplication_cpu(a, b, size)
    assert_same_values(c, a)

    a_transposed_transposed = transposed(transposed(a, size), size)
    assert_same_values(a_transposed_transposed, a) 
elsif mode == "CPU"
    puts "CPU Computation, wait..."
    c = matrix_multiplication_cpu(a, b, size)
    # Let's just assume that this works if CHECK passes
elsif mode == "GPU"
    puts "GPU Computation"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Running computation..."
    c_gpu = matrix_multiplication_gpu(a, b, size)
    c_gpu[0]    # Access an element to ensure that the computation started
    c_cpu = matrix_multiplication_cpu(a, b, size)

    assert_same_values(c_gpu, c_cpu)
elsif mode == "GPU_T_A"
    # Transpose A
    a_transposed = transposed(a, size)

    puts "GPU Computation"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Running computation..."
    c_gpu = matrix_multiplication_gpu_transposed_a(a_transposed, b, size)
    c_gpu[0]    # Access an element to ensure that the computation started
    c_cpu = matrix_multiplication_cpu(a, b, size)

    assert_same_values(c_gpu, c_cpu)
elsif mode == "GPU_T_B"
    b_transposed = transposed(b, size)

    puts "GPU Computation"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Running computation..."
    c_gpu = matrix_multiplication_gpu_transposed_b(a, b, size)
    c_gpu[0]    # Access an element to ensure that the computation started
    c_cpu = matrix_multiplication_cpu(a, b, size)

    assert_same_values(c_gpu, c_cpu)
elsif mode == "GPU_BLOCK"
    puts "GPU Computation"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Running computation..."
    c_gpu = matrix_multiplication_gpu_blocked(a, b, size)
    c_cpu = matrix_multiplication_cpu(a, b, size)

    assert_same_values(c_gpu, c_cpu)
else
    puts "Invalid rendering mode: Pass one of CHECK, CPU, GPU, GPU_T_A, GPU_T_B or GPU_BLOCK as argument"
end
