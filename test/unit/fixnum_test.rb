require "ikra"
require_relative "unit_test_template"

class FixnumTest < UnitTestCase
    def test_abs
        array = Array.pnew(100) do |j|
            1.abs() + 1.abs() + 0
        end

        assert_equal(200, array.reduce(:+))
    end

    def test_bit_length
        array = Array.pnew(100) do |j|
            0x100.bit_length()
        end

        assert_equal(900, array.reduce(:+))
    end

    def test_div
        array = Array.pnew(100) do |j|
            9.div(4)
        end

        assert_equal(200, array.reduce(:+))
    end

    def test_even
        array = Array.pnew(100) do |j|
            if j.even? then 1 else 0 end
        end

        assert_equal(50, array.reduce(:+))
    end

    def test_fdiv
        array = Array.pnew(100) do |j|
            5.fdiv(2) + 5.fdiv(2.5)
        end

        assert_equal(450, array.reduce(:+))
    end

    def test_magnitude
        array = Array.pnew(100) do |j|
            -1.magnitude() + 1.magnitude() + 0
        end

        assert_equal(200, array.reduce(:+))
    end

    def test_modulo
        array = Array.pnew(100) do |j|
            234.modulo(5)
        end

        assert_equal(400, array.reduce(:+))
    end

    def test_odd
        array = Array.pnew(100) do |j|
            if j.odd? then 1 else 0 end
        end

        assert_equal(50, array.reduce(:+))
    end

    def test_pow
        array = Array.pnew(100) do |j|
            j/4 ** 2
        end

        assert_equal(264, array.reduce(:+))
        assert_equal(::Fixnum, array[0].class)
    end

    def test_size
        array = Array.pnew(100) do |j|
            2.size
        end

        assert_in_delta(100 * 6, array.reduce(:+), 200)
    end

    def test_next
        array = Array.pnew(100) do |j|
            -10.next
        end

        assert_equal(-900, array.reduce(:+))
    end


    def test_succ
        array = Array.pnew(100) do |j|
            -10.succ
        end

        assert_equal(-900, array.reduce(:+))
    end


    def test_zero
        array = Array.pnew(100) do |j|
            if j.zero? then 1 else 0 end
        end

        assert_equal(1, array.reduce(:+))
    end


    def test_bitor
        array = Array.pnew(100) do |j|
            12 | 20
        end

        assert_equal(100 * 28, array.reduce(:+))
    end


end
