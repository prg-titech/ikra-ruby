require "ikra"
require_relative "unit_test_template"

class LexicalVariableTest < UnitTestCase
    def test_simple_lexical_variable_int
        var1 = 10

        base_array = Array.pnew(100) do |j|
            var1
        end

        assert_equal(1000, base_array.reduce(:+))
    end

    def test_simple_lexical_variable_float
        var1 = 10.0

        base_array = Array.pnew(100) do |j|
            var1
        end

        assert_in_delta(1000.0, base_array.reduce(:+), 0.001)
    end
end