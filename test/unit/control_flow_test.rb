require "ikra"
require_relative "unit_test_template"

class ControlFlowTest < UnitTestCase
    def test_if
        half_ones = Array.pnew(100) do |j|
            if j % 2 == 0
                return 1
            else
                return 0
            end
        end

        assert_equal(50, half_ones.reduce(:+))
    end

    def test_for
        sum = Array.pnew(100) do |j|
            result = 0
            for i in 1..j
                result += i
            end

            result
        end

        for k in 0..99
            assert_equal(k*(k+1) / 2, sum[k])
        end
    end

    def test_while
        sum = Array.pnew(100) do |j|
            result = 0
            i = 1

            while i <= j
                result += i
                i += 1
            end

            result
        end

        for k in 0..99
            assert_equal(k*(k+1) / 2, sum[k])
        end
    end
end