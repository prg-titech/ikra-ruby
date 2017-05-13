require "ikra"

def generate_matrix(size, seed)
    return Array.new(size * size) do |i|
        (i * 17 + 6) % (seed + 27)
    end
end

def generate_identity_matrix(size)
    return Array.new(size * size) do |i|
        if i / size == i % size
            1
        else
            0
        end
    end
end

def matrix_multiplication_cpu(a, b, size)
    return Array.new(size * size) do |index|
        x = index % size
        y = index / size
        result = 0

        for i in 0...size
            result = result + a[y * size + i] * b[i * size + x]
        end

        result
    end
end

def matrix_multiplication_gpu(a, b, size)
    return PArray.new(size * size, block_size: 512) do |index|
        x = index % size
        y = index / size
        result = 0

        for i in 0...size
            result = result + a[y * size + i] * b[i * size + x]
        end

        result
    end
end

def matrix_sanity_check_cpu
    size = 10
    a = generate_matrix(size, 0)
    b = generate_identity_matrix(size)

    c = matrix_multiplication_cpu(a, b, size)

    for i in 0...(size*size)
        if a[i] != c[i]
            raise "Sanity check failed in field #{i}: Found #{c[i]} but expected #{a[i]}"
        end
    end
end

def matrix_sanity_check_gpu
    size = 10
    a = generate_matrix(size, 0)
    b = generate_identity_matrix(size)

    c = matrix_multiplication_gpu(a, b, size)

    for i in 0...(size*size)
        if a[i] != c[i]
            raise "Sanity check failed in field #{i}: Found #{c[i]} but expected #{a[i]}"
        end
    end
end


def matrix_gpu
    size = 75
    a = generate_matrix(size, 0)
    b = generate_matrix(size, 2)

    c1 = matrix_multiplication_gpu(a, b, size)

    Ikra::Log.info("Running CPU version to compare result")
    time_before = Time.now
    c2 = matrix_multiplication_cpu(a, b, size)
    Ikra::Log.info("Done, took #{Time.now - time_before} s on CPU")

    for i in 0...(size*size)
        if c1[i] != c2[i]
            raise "Matrix multiply failed in field #{i}: Found #{c1[i]} but expected #{c2[i]}"
        end
    end
end