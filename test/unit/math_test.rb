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

    def test_acosh
        array = Array.pnew(100) do |j|
            Math.acosh(5) + 0
        end

        assert_in_delta(100*2.29243166956, array.reduce(:+), 0.01)
    end

    def test_asin
        array = Array.pnew(100) do |j|
            Math.asin(0.3) + 0
        end

        assert_in_delta(100*0.304693, array.reduce(:+), 0.01)
    end

    def test_asinh
        array = Array.pnew(100) do |j|
            Math.asinh(5) + 0
        end

        assert_in_delta(100*2.31243834, array.reduce(:+), 0.01)
    end

    def test_atan
        array = Array.pnew(100) do |j|
            Math.atan(0.3) + 0
        end

        assert_in_delta(100*0.291456, array.reduce(:+), 0.01)
    end

    def test_atan2
        array = Array.pnew(100) do |j|
            Math.atan2(0.2, 0.5) + 0
        end

        assert_in_delta(100*0.380506, array.reduce(:+), 0.01)
    end

    def test_atanh
        array = Array.pnew(100) do |j|
            Math.atanh(0.5) + 0
        end

        assert_in_delta(100*0.549306, array.reduce(:+), 0.01)
    end

    def test_cbrt
        array = Array.pnew(100) do |j|
            Math.cbrt(0.3) + 0
        end

        assert_in_delta(100*0.669433, array.reduce(:+), 0.01)
    end

    def test_cos
        array = Array.pnew(100) do |j|
            Math.cos(2) + 0
        end

        assert_in_delta(100*-0.41614683, array.reduce(:+), 0.01)
    end

    def test_cosh
        array = Array.pnew(100) do |j|
            Math.cosh(0.3) + 0
        end

        assert_in_delta(100*1.045339, array.reduce(:+), 0.01)
    end

    def test_erf
        array = Array.pnew(100) do |j|
            Math.erf(0.3) + 0
        end

        assert_in_delta(100*0.328627, array.reduce(:+), 0.01)
    end

    def test_erfc
        array = Array.pnew(100) do |j|
            Math.erfc(0.3) + 0
        end

        assert_in_delta(100*0.671373, array.reduce(:+), 0.01)
    end

    def test_exp
        array = Array.pnew(100) do |j|
            Math.exp(0.3) + 0
        end

        assert_in_delta(100*1.349859, array.reduce(:+), 0.01)
    end

    def test_gamma
        array = Array.pnew(100) do |j|
            Math.gamma(0.3) + 0
        end

        assert_in_delta(100*2.99157, array.reduce(:+), 0.01)
    end

    def test_hypot
        array = Array.pnew(100) do |j|
            Math.hypot(0.3, 0.5) + 0
        end

        assert_in_delta(100*0.583095, array.reduce(:+), 0.01)
    end

    def test_ldexp
        array = Array.pnew(100) do |j|
            Math.ldexp(0.3, 5) + 0
        end

        assert_in_delta(100*9.6, array.reduce(:+), 0.01)
    end

    def test_lgamma
        array = Array.pnew(100) do |j|
            Math.lgamma(0.3) + 0
        end

        assert_in_delta(100*1.09580, array.reduce(:+), 0.01)
    end

    def test_log
        array = Array.pnew(100) do |j|
            Math.log(0.3) + 0
        end

        assert_in_delta(100*-1.20397, array.reduce(:+), 0.01)
    end

    def test_log10
        array = Array.pnew(100) do |j|
            Math.log10(0.3) + 0
        end

        assert_in_delta(100*-0.522879, array.reduce(:+), 0.01)
    end

    def test_log2
        array = Array.pnew(100) do |j|
            Math.log2(0.3) + 0
        end

        assert_in_delta(100*-1.73697, array.reduce(:+), 0.01)
    end

    def test_sin
        array = Array.pnew(100) do |j|
            Math.sin(0.3) + 0
        end

        assert_in_delta(100*0.295520, array.reduce(:+), 0.01)
    end

    def test_sinh
        array = Array.pnew(100) do |j|
            Math.sinh(5) + 0
        end

        assert_in_delta(100*74.2032105, array.reduce(:+), 0.01)
    end

    def test_sqrt
        array = Array.pnew(100) do |j|
            Math.sqrt(0.3) + 0
        end

        assert_in_delta(100*0.547723, array.reduce(:+), 0.01)
    end

    def test_tan
        array = Array.pnew(100) do |j|
            Math.tan(0.3) + 0
        end

        assert_in_delta(100*0.309336, array.reduce(:+), 0.01)
    end

    def test_tanh
        array = Array.pnew(100) do |j|
            Math.tanh(0.3) + 0
        end

        assert_in_delta(100*0.291313, array.reduce(:+), 0.01)
    end
end
