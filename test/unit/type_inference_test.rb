require "ikra"
require_relative "unit_test_template"

class TypeInferenceTest < UnitTestCase
    def test_simple_type_inference
        # Only one (linear) pass required
        array = Array.pnew(100) do |j|
            x1 = j * 2
            x2 = x1 % 2 == 0

            if x2
                x1
            else
                0
            end
        end

        assert_equal(9900, array.reduce(:+))
    end

    def test_type_inference_assign_int_float
        # Only one (linear) pass required
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = j.to_f
            end

            x1
        end

        assert_in_delta(4950, array.reduce(:+), 0.001)

        for index in 0...100
            if index % 2 == 0
                expected_type = ::Float
            else
                expected_type = ::Fixnum
            end

            assert_equal(array[index].class, expected_type)
        end
    end

    def test_union_type_primitive_invocation
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            x1 * 2
        end

        assert_in_delta(7400 * 2, array.reduce(:+), 0.001)

        for index in 0...100
            if index % 2 == 0
                expected_type = ::Float
            else
                expected_type = ::Fixnum
            end

            assert_equal(array[index].class, expected_type)
        end
    end

    def test_type_inference_multiple_passes
        array = Array.pnew(100) do |j|
            x = 0

            for i in 0..10
                x = x + 2.01
                x = x.to_f

                if i == 5
                    # Floor
                    x = x.to_i
                end
            end

            x
        end

        assert_in_delta(2205, array.reduce(:+), 0.001)
    end


    def test_nil
        array = Array.pnew(100) do |j|
            nil
        end

        assert_equal(nil, array[0])
    end


    def test_type_inference_assign_int_nil
        # Only one (linear) pass required
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = nil
            end

            x1
        end

        for index in 0...100
            if index % 2 == 0
                expected_type = ::NilClass
            else
                expected_type = ::Fixnum
            end

            assert_equal(array[index].class, expected_type)
        end
    end
end
