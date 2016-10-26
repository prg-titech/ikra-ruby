require "ikra"
require_relative "unit_test_template"

class TypeInferenceTest < UnitTestCase
    def test_simple_type_inference
        # Only one (linear) pass required
        base_array = Array.pnew(100) do |j|
            x1 = j * 2
            x2 = x1 % 2 == 0

            if x2
                x1
            else
                0
            end
        end

        assert_equal(9900, base_array.reduce(:+))
    end

    def test_type_inference_assign_int_float
        # Only one (linear) pass required
        base_array = Array.pnew(100) do |j|
            x1 = 1
            x1 = 1.0
            x1
        end

        assert_in_delta(100.0, base_array.reduce(:+), 0.001)
    end
end