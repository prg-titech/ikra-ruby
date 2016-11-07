require "ikra"
require_relative "unit_test_template"

class ArrayTest < UnitTestCase
    def test_lexical_int_array
        lex_array = (1..100).to_a

        partial_sums = Array.pnew(100) do |j|
            result = 0

            for i in 0...j
                result = result + lex_array[i]
            end

            result
        end

        assert_equal(166650, partial_sums.reduce(:+))
    end

    def test_lexical_float_array
        lex_array = (1..100).map do |i| i.to_f + 0.1 end

        partial_sums = Array.pnew(100) do |j|
            result = 0

            for i in 0...j
                result = result + lex_array[i]
            end

            result
        end

        assert_in_delta(167145.0, partial_sums.reduce(:+), 0.1)
    end

    # TODO: Fix polymorphic arguments
    def test_lexical_union_array
        lex_array = (1..100).map do |i|
            if (i % 2 == 0)
                i
            else
                i.to_f + 100.0
            end
        end

        partial_sums = Array.pnew(100) do |j|
            result = 0.0

            for i in 0...j
                result = result + lex_array[i]
            end

            result
        end

        assert_in_delta(416650.0, partial_sums.reduce(:+), 0.1)
    end
end
