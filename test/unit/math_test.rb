require "ikra"
require_relative "unit_test_template"

class MathTest < UnitTestCase
    def test_trigonometric
        array = Array.pnew(100) do |j|
            Math.cos(3.14) + Math.sin(0) + 0
        end

        assert_in_delta(-100, array.reduce(:+), 0.01)
    end

    def test_acos
        array = Array.pnew(100) do |j|
            Math.acos(0.3) + 0
        end

        assert_in_delta(100*1.266104, array.reduce(:+), 0.01)
    end
end
