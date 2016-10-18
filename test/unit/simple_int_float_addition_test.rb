require "ikra"
require_relative "unit_test_template"

class SimpleIntFloatAdditionTest < UnitTestCase
    def test_kernel_float_plus_int
        all_floats = Array.pnew(100) do |j|
            1.12 + j
        end

        for i in 0..99
            assert_in_delta(1.12 + i, all_floats[i], 0.001)
        end
    end

    def test_kernel_int_plus_float
        all_floats = Array.pnew(100) do |j|
            j + 1.12
        end

        for i in 0..99
            assert_in_delta(1.12 + i, all_floats[i], 0.001)
        end
    end
end