require "ikra"
require_relative "unit_test_template"

class ReduceTest < UnitTestCase
    def test_reduce
        array = PArray.new(511) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_shortcut
        array = PArray.new(511) do |j|
            j+1
        end

        result1 = array.reduce(:+)

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce2
        array = PArray.new(4096) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce3
        array = PArray.new(513) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce4
        array = PArray.new(514) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce5
        array = PArray.new(517) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_single
        array = PArray.new(1) do |j|
            j+123
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_high
        array = PArray.new(25477) do |j|
            (j+1)%200
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_two_pow_twelve
        array = PArray.new(4096) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_zero
        array = PArray.new(0) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_two
        array = PArray.new(2) do |j|
            j+1
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_equal(result2 , result1[0])
    end

    def test_reduce_thousand_floats
        array = PArray.new(1024) do |j|
            j.fdiv(3)
        end

        result1 = array.reduce do |l, r|
            l + r
        end

        result2 = array.to_a.reduce do |l, r|
            l + r
        end

        assert_in_delta(result2 , result1[0], 0.00001)
    end


    def test_reduce_after_combine
        array1 = PArray.new(2577) do |j|
            (j+1)%200
        end

        array2 = PArray.new(2577) do |j|
            (j+1)%200
        end

        temp_combine = array1.combine(array2) do |a, b|
            a + b
        end

        result = temp_combine.reduce do |l, r|
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

end
