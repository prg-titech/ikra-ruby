require "ikra"
require_relative "unit_test_template"

class SimpleMapKernelTest < UnitTestCase
    def test_kernel
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
end