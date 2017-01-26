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
            assert_equal(array[index].class, ::Float)
        end
    end

    def test_coerce_type_primitive_invocation
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = 2 * j.to_f + 0.1
            end

            x1 * 2
        end

        assert_in_delta(14810.0, array.reduce(:+), 0.001)

        for index in 0...100
            assert_equal(array[index].class, ::Float)
        end
    end

    def test_coerce_type_primitive_invocation_2
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            # Inversed
            2 * x1
        end

        assert_in_delta(7400 * 2, array.reduce(:+), 0.001)

        for index in 0...100
            assert_equal(array[index].class, ::Float)
        end
    end

    def test_int_method_gg
        array = Array.pnew(100) do |j|
            x1 = 0

            if j%2 == 0
                x1 = 1
            else
                x1 = 0
            end

            # Inversed
            2 >> x1
        end

        assert_in_delta(150, array.reduce(:+), 0.001)

        for index in 0...100
            assert_equal(array[index].class, ::Fixnum)
        end
    end

    def test_coerce_type_primitive_invocation_4
        array = Array.pnew(100) do |j|
            x1 = j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            # Inversed
            2.1 * x1
        end

        assert_in_delta(15540.0, array.reduce(:+), 0.001)

        for index in 0...100
            assert_equal(array[index].class, ::Float)
        end
    end

    def test_coerce_type_primitive_invocation_5
        array = Array.pnew(100) do |j|
            x1 = 2 * j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            result = 0
            if x1 >= j
                result = result + 1
            end

            result
        end

        assert_equal(100, array.reduce(:+))
    end

    def test_coerce_type_primitive_invocation_6
        array = Array.pnew(100) do |j|
            x1 = 2 * j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            result = 0
            if j <= x1
                result = result + 1
            end

            result
        end

        assert_equal(100, array.reduce(:+))
    end

    def test_coerce_type_primitive_invocation_7
        array = Array.pnew(100) do |j|
            x1 = 2 * j

            if j%2 == 0
                x1 = 2 * j.to_f
            end

            result = 0
            if j.to_f <= x1
                result = result + 1
            end

            result
        end

        assert_equal(100, array.reduce(:+))
    end

    def test_coerce_type_primitive_invocation_8
        array = Array.pnew(100) do |j|
            x1 = j
            x2 = j

            if j % 4 == 0
               # INT x INT
            elsif j % 4 == 1
                x1 = j.to_f + 0.1
            elsif j % 4 == 2
                x2 = j.to_f + 0.2
            elsif j % 4 == 3
                x1 = j.to_f + 0.3
                x2 = j.to_f + 0.4
            end

            # Inversed
            x1 * x2
        end

        assert_in_delta(329618.0, array.reduce(:+), 0.001)

        for index in 0...100
            assert_equal(array[index].class, ::Float)
        end
    end

    def test_coerce_type_primitive_invocation_9
        # Invoke method

        array = Array.pnew(100) do |j|
            x1 = 3

            if j%2 == 0
                x1 = 3.0
            end

            Math.ldexp(x1, 4.5)
        end

        assert_in_delta(48.0 * 100, array.reduce(:+), 0.001)
    end

    def test_coerce_type_primitive_invocation_10
        # Invoke method

        array = Array.pnew(100) do |j|
            x1 = j
            x2 = 4.5

            if j%2 == 0
                x1 = j.to_f + 0.1
                x2 = 4
            end

            Math.ldexp(x1, x2)
        end

        assert_in_delta(79280.0, array.reduce(:+), 0.001)
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

    def test_union_type_method_invocation
        # Returns union type

        array = Array.pnew(100) do |j|
            x1 = true
            x2 = true

            if j % 2 == 0
                x1 = j
                x2 = j
            end

            x3 = x1 & x2

            x3
        end

        expected = Array.new(100) do |j|
            if j % 2 == 0
                j
            else
                true
            end
        end

        assert_equal(array.to_a, expected)
    end

    def test_union_type_method_invocation_without_args
        Ikra::RubyIntegration.implement(Ikra::RubyIntegration::INT_S, :dummy_method, Ikra::RubyIntegration::INT, 0, "(100)")
        Ikra::RubyIntegration.implement(Ikra::RubyIntegration::BOOL_S, :dummy_method, Ikra::RubyIntegration::INT, 0, "(200)")

        array = Array.pnew(100) do |j|
            x1 = true
            
            if j % 2 == 0
                x1 = 1
            end

            x1.dummy_method
        end

        expected = Array.new(100) do |j|
            if j % 2 == 0
                100
            else
                200
            end
        end

        assert_equal(array.to_a, expected)
    end
end
