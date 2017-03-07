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

    def test_lexical_array_in_host_section
        a1 = Array.pnew(511) do |j|
            j + 1
        end

        lex_array = (100...200).to_a

        section_result = Ikra::Symbolic.host_section(a1) do |a1|
            a1.pmap(with_index: true) do |k, i|
                k + 1 + lex_array[i % 100]
            end
        end

        result = Array.new(511) do |j|
            j + 2 + lex_array[j % 100]
        end

        assert_equal(result, section_result.to_a)
    end
end
