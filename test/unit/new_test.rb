require "ikra"
require_relative "unit_test_template"

class NewTest < UnitTestCase
    def test_new
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1
        end

        assert_equal(array_cpu.reduce(:+) , array_gpu.reduce(:+))
    end

    def test_new_2d
        array_gpu = Array.pnew(dimensions: [2, 3]) do |x, y|
            x * 100 + y
        end

        assert_equal([0, 1, 2, 100, 101, 102] , array_gpu.to_a)
    end

    def test_new_3d
        array_gpu = Array.pnew(dimensions: [2, 3, 4]) do |x, y, z|
            x * 100 + y * 10 + z
        end

        assert_equal([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123] , array_gpu.to_a)
    end
end
