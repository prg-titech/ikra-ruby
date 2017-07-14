require "ikra"
require_relative "unit_test_template"

class NewTest < UnitTestCase
    def test_new
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1
        end

        assert_equal(array_cpu.reduce(:+) , array_gpu.to_a.reduce(:+))
    end

    def test_new_2d
        array_gpu = PArray.new(dimensions: [2, 3]) do |index|
            index[0] * 100 + index[1]
        end

        assert_equal([0, 1, 2, 100, 101, 102] , array_gpu.to_a)
    end

    def test_new_3d
        array_gpu = PArray.new(dimensions: [2, 3, 4]) do |index|
            index[0] * 100 + index[1] * 10 + index[2]
        end

        assert_equal([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123] , array_gpu.to_a)
    end
end
