require "ikra"
require_relative "unit_test_template"

class KernelFusionTest < UnitTestCase
    def test_fusion_2
        base_array = PArray.new(100) do |j|
            j + 1
        end

        mapped_array = base_array.map do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1), mapped_array[i])
        end
    end

    def test_fusion_3
        base_array = PArray.new(100) do |j|
            j + 1
        end

        mapped_array = base_array.map do |j|
            j * j
        end

        mapped_array = mapped_array.map do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1) * (i + 1) * (i + 1), mapped_array[i])
        end
    end

    def test_reuse_computation
        # keep, cached, keep_previous, cache_previous, preserve_input
        # base_array.keep.map
        # Ikra.require(mapped_array1, mapped_array2)
        
        base_array = PArray.new(100, keep: true) do |j|
            j + 1
        end

        mapped_array1 = base_array.map do |j|
            j * j
        end

        mapped_array2 = base_array.map do |j|
            j * j
        end

        assert_equal(mapped_array1.to_a.reduce(:+), mapped_array2.to_a.reduce(:+))
    end

    def test_reuse_computation_2
        # keep, cached, keep_previous, cache_previous, preserve_input
        # base_array.keep.map
        # Ikra.require(mapped_array1, mapped_array2)
        
        base_array = PArray.new(100, keep: true) do |j|
            j + 1
        end

        mapped_array1 = base_array.map(keep: true) do |j|
            j * j
        end

        mapped_array2 = mapped_array1.map(keep: false) do |j|
            j + 9
        end
        
        mapped_array3 = mapped_array1.map(keep: false) do |j|
            j + 9
        end

        # EXPCTED, FOUND
        assert_equal(mapped_array2.to_a, mapped_array3.to_a)
    end

    def test_stencil
        # GPU computation
        base_array_gpu = PArray.new(100, keep: true) do |j|
            j + 100
        end


        stencil_result_gpu = base_array_gpu.stencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu = stencil_result_gpu.to_a.reduce(:+)


        stencil_result_gpu2 = base_array_gpu.stencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu2 = stencil_result_gpu2.to_a.reduce(:+)


        # Compare results
        assert_equal(aggregated_gpu, aggregated_gpu2)
    end

   def test_iterative_stencil
        # GPU computation
        p = PArray.new(100) do |j|
            j
        end

        for i in 0...10
            p = p.stencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
                p0 - p1 + p2
            end 
        end

        aggregated_gpu = p.to_a.reduce(:+)


        q = Array.new(100) do |j|
            j
        end

        for i in 0...10
            q = q.stencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
                p0 - p1 + p2
            end
        end

        aggregated_cpu = q.reduce(:+)


        # Compare results
        assert_equal(aggregated_gpu, aggregated_cpu)
    end

   def test_iterative_map_with_reduce
        # GPU computation
        p = PArray.new(100) do |j|
            j
        end

        while p.to_a.reduce(:+) < 5400
            p = p.map do |i| i + 1 end
        end


        # CPU computation
        q = Array.new(100) do |j|
            j
        end

        while q.reduce(:+) < 5400
            q = q.map do |i| i + 1 end
        end


        # Compare results
        assert_equal(p.to_a, q)
    end

   def test_iterative_map_with_reduce_and_keep
        # GPU computation
        p = PArray.new(100) do |j|
            j
        end

        while p.to_a.reduce(:+) < 5400
            p = p.map(keep: true) do |i| i + 1 end
        end


        # CPU computation
        q = Array.new(100) do |j|
            j
        end

        while q.reduce(:+) < 5400
            q = q.map do |i| i + 1 end
        end


        # Compare results
        assert_equal(p.to_a, q)
    end
end