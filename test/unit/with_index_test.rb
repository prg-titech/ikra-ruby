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
end
