require "ikra"
require_relative "unit_test_template"

class ReduceTest < UnitTestCase
    def test_reduce
        array = Array.pnew(511) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce2
        array = Array.pnew(4096) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce3
        array = Array.pnew(513) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce4
        array = Array.pnew(514) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce5
        array = Array.pnew(517) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_single
        array = Array.pnew(1) do |j|
            j+123
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_high
        array = Array.pnew(25477) do |j|
            (j+1)%200
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_two_pow_twelve
        array = Array.pnew(4096) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_zero
        array = Array.pnew(0) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_two
        array = Array.pnew(2) do |j|
            j+1
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_thousand_floats
        array = Array.pnew(1024) do |j|
            j.fdiv(3)
        end

        result1 = array.preduce do |l, r| 
            l + r
        end

        result2 = array.reduce do |l, r| 
            l + r
        end

        assert_in_delta(result2 , result1[0], 0.00001)
    end

"""
    def test_reduce_after_combine
        array1 = Array.pnew(2577) do |j|
            (j+1)%200
        end

        array2 = Array.pnew(2577) do |j|
            (j+1)%200
        end

        temp_combine = array1.pcombine(array2) do |a, b|
            a + b
        end

        result = temp_combine.preduce do |l, r| 
            l + r
        end


        array_cpu1 = Array.new(2577) do |j|
            (j+1)%200
        end

        array_cpu2 = Array.new(2577) do |j|
            (j+1)%200
        end

        temp_combine_cpu = array_cpu1.combine(array_cpu2) do |a, b|
            a + b
        end

        result_cpu = temp_combine_cpu.reduce do |l, r| 
            l + r
        end

        assert_equal(result_cpu , result[0])
    end
"""
end
