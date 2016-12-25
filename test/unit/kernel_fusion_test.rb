require "ikra"
require_relative "unit_test_template"

class KernelFusionTest < UnitTestCase
    def test_fusion_2
        base_array = Array.pnew(100) do |j|
            j + 1
        end

        mapped_array = base_array.pmap do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1), mapped_array[i])
        end
    end

    def test_fusion_3
        base_array = Array.pnew(100) do |j|
            j + 1
        end

        mapped_array = base_array.pmap do |j|
            j * j
        end

        mapped_array = mapped_array.pmap do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1) * (i + 1) * (i + 1), mapped_array[i])
        end
    end

    def test_reuse_computation
        # keep, cached, keep_previous, cache_previous, preserve_input
        # base_array.keep.pmap
        # Ikra.require(mapped_array1, mapped_array2)
        
        base_array = Array.pnew(100, keep: true) do |j|
            j + 1
        end

        mapped_array1 = base_array.pmap do |j|
            j * j
        end

        mapped_array2 = base_array.pmap do |j|
            j * j
        end

        assert_equal(mapped_array1.reduce(:+), mapped_array2.reduce(:+))
    end



    def test_stencil
        # GPU computation
        base_array_gpu = Array.pnew(100, keep: true) do |j|
            j + 100
        end


        stencil_result_gpu = base_array_gpu.pstencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        stencil_result_gpu2 = base_array_gpu.pstencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu2 = stencil_result_gpu2.reduce(:+)


        # Compare results
        assert_equal(aggregated_gpu, aggregated_gpu2)
    end
end