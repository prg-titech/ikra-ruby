require "ikra"
require_relative "unit_test_template"

class BoolTest < UnitTestCase
    def test_bool
        all_zeroes = PArray.new(100) do |j|
            true
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_union_bool
        all_zeroes = PArray.new(100) do |j|
            if j % 2 == 0
                x = true
            else
                x = 15
            end
            x
        end

        for i in 0..99
            assert_equal(i % 2 == 0 ? true : 15 , all_zeroes[i])
        end
    end

    def test_and1
        all_zeroes = PArray.new(100) do |j|
            true and false
        end

        for i in 0..99
            assert_equal(false , all_zeroes[i])
        end
    end

    def test_and2
        all_zeroes = PArray.new(100) do |j|
            true && true
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_and3
        all_zeroes = PArray.new(100) do |j|
            true & false
        end

        for i in 0..99
            assert_equal(false , all_zeroes[i])
        end
    end

    def test_and4
        all_zeroes = PArray.new(100) do |j|
            true & true
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_xor1
        all_zeroes = PArray.new(100) do |j|
            true ^ false
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_xor2
        all_zeroes = PArray.new(100) do |j|
            true ^ true
        end

        for i in 0..99
            assert_equal(false , all_zeroes[i])
        end
    end

    def test_or1
        all_zeroes = PArray.new(100) do |j|
            true | false
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_or2
        all_zeroes = PArray.new(100) do |j|
            false | false
        end

        for i in 0..99
            assert_equal(false , all_zeroes[i])
        end
    end

    def test_or3
        all_zeroes = PArray.new(100) do |j|
            true || true
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_or4
        all_zeroes = PArray.new(100) do |j|
            false || false
        end

        for i in 0..99
            assert_equal(false , all_zeroes[i])
        end
    end

    def test_or5
        all_zeroes = PArray.new(100) do |j|
            true || false
        end

        for i in 0..99
            assert_equal(true , all_zeroes[i])
        end
    end

    def test_smaller
        arr1 = PArray.new(100) do |j|
            j
        end

        arr2 = PArray.new(100) do |j|
            100 - j
        end

        result_cpu = Array.new(100) do |j|
            j < (100 - j)
        end

        assert_equal(result_cpu, (arr1 < arr2).to_a)
    end

    def test_smaller_float
        arr1 = PArray.new(100) do |j|
            j.to_f
        end

        arr2 = PArray.new(100) do |j|
            100 - j.to_f
        end

        result_cpu = Array.new(100) do |j|
            j < (100 - j)
        end

        assert_equal(result_cpu, (arr1 < arr2).to_a)
    end
end
