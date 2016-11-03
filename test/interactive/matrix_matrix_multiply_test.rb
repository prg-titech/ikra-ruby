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
    # TODO: Implement
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

        # TODO: Implement

        123.4
    end
end

# GPU-based implementation
def matrix_multiplication_gpu(a, b, size)
    return Array.pnew(size * size) do |index|
        x = index % size
        y = index / size

        # TODO: Implement

        123.4
    end
end

# GPU-based implementation with blocking
def matrix_multiplication_gpu_blocked(a, b, size)
    block_size_x = 15
    block_size_y = 15

    # Create blocked indices
    indices = Array.new(size * size)

    # TODO: Generate block indices

    unordered_result = indices.pmap do |index|
        x = index % size
        y = index / size

        # TODO: Implement

        123.4
    end

    correct_order_result = Array.new(size*size)

    # TODO: Restore correct order

    return correct_order_result
end

# GPU-based implementation where b is transposed
def matrix_multiplication_gpu_transposed_b(a, b, size)
    # Matrix B is transposed

    return Array.pnew(size * size) do |index|
        x = index % size
        y = index / size

        # TODO: Implement

        123.4
    end
end

# GPU-based implementation where a is transposed
def matrix_multiplication_gpu_transposed_a(a, b, size)
    # Matrix A is transposed

    return Array.pnew(size * size) do |index|
        x = index % size
        y = index / size

        # TODO: Implement

        123.4
    end
end

mode = ARGV[0]
size = ARGV[1].to_i

if size % 150 != 0
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

    a = transposed(transposed(a, size), size)
    c = matrix_multiplication_cpu(a, b, size)
    assert_same_values(c, a) 
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
