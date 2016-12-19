require "ikra"
require_relative "unit_test_template"

class RegressionTest < UnitTestCase
    def test_explicit_return
        all_ones = Array.pnew(100) do |j|
            return 1
        end

        assert_equal(100, all_ones.reduce(:+))
    end

    def test_lexical_scope_variable_redefinition
        var = 1

        id_kernel = Array.pnew(100) do |var|
            var
        end

        assert_equal(100, id_kernel.reduce(:+))
    end

    def test_modulus_float_int
        array = Array.pnew(100) do |var|
            4.0 % 3
        end

        assert_in_delta(100.0, array.reduce(:+), 0.01)
    end

    def test_basic_array_operations_without_ikra
        a1 = [1, 2, 3]
        a2 = ["a", "b", "c"]
        r = a1 + a2

        assert_equal(1, r[0])
        assert_equal("a", r[3])
    end
end
