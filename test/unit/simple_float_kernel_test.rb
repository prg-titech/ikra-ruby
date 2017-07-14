require "ikra"
require_relative "unit_test_template"

class SimpleFloatKernelTest < UnitTestCase
    def test_kernel
        all_floats = PArray.new(100) do |j|
            1.12
        end

        for i in 0..99
            assert_in_delta(1.12, all_floats[i], 0.001)
        end
    end
end