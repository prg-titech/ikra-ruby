require "ikra"
require_relative "unit_test_template"

class WithIndexTest < UnitTestCase
    def test_map
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        result_gpu = array_gpu.pmap.with_index do |value, index|
            value + (index % 10)
        end

        array_cpu = Array.new(511) do |j|
            j + 1
        end

        result_cpu = array_cpu.map.with_index do |value, index|
            value + (index % 10)
        end

        assert_equal(result_cpu, result_gpu.to_a)
    end


    def test_map_2d
        array_gpu = Array.pnew(dimensions: [2, 3]) do |index|
            index[0] * 10 + index[1]
        end

        result_gpu = array_gpu.pmap.with_index do |value, index|
            value + 1000 * index[0] + 100 * index[1]
        end

        assert_equal([0, 101, 202, 1010, 1111, 1212] , result_gpu.to_a)
    end

    def test_stencil_2d
        # GPU computation
        base_array_gpu = Array.pnew(dimensions: [7, 5]) do |index|
            10 * index[0] + index[1]
        end

        stencil_result_gpu = base_array_gpu.pstencil([[-1, -1], [0, -1], [0, 1], [0, 0]], 10000).with_index do |values, index|
            # values[-1, -1] + values[0, -1]
            values[0] + values[1] + values[2] + values[3] + 1000 * index[0] + 100 * index[1]
        end

        # Compare results
        assert_equal([
            10000, 10000, 10000, 10000, 10000, 
            10000, 1133, 1237, 1341, 10000, 
            10000, 2173, 2277, 2381, 10000, 
            10000, 3213, 3317, 3421, 10000, 
            10000, 4253, 4357, 4461, 10000, 
            10000, 5293, 5397, 5501, 10000, 
            10000, 6333, 6437, 6541, 10000], stencil_result_gpu.to_a)
    end
end
