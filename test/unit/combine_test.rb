require "ikra"
require_relative "unit_test_template"

class CombineTest < UnitTestCase
    def test_combine
        arrayA = PArray.new(101) do |j|
            j
        end

        arrayB = PArray.new(101) do |j|
            5
        end

        arrayC = PArray.new(101) do |j|
            j * 2
        end
        
        arrayA = arrayA.combine(arrayB, arrayC) do |a,b,c|
            a + b + c
        end

        assert_equal(15655, arrayA.to_a.reduce(:+))
    end


    def test_combine_no_arg
        arrayA = PArray.new(101) do |j|
            j
        end
        
        arrayA = arrayA.combine do |a|
            a
        end

        assert_equal(5050, arrayA.to_a.reduce(:+))
    end

    def test_combine_three_arg
        arrayA = PArray.new(101) do |j|
            j
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        arrayC = PArray.new(101) do |j|
            j*3
        end
        arrayD = PArray.new(101) do |j|
            j*4
        end
        
        arrayA = arrayA.combine(arrayB, arrayC, arrayD) do |a,b,c,d|
            a+b+c+d
        end

        assert_equal(50500, arrayA.to_a.reduce(:+))
    end

    def test_combine_plus
        arrayA = PArray.new(101) do |j|
            j
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        
        arrayA = arrayA + arrayB

        assert_equal(15150, arrayA.to_a.reduce(:+))
    end

    def test_combine_plus_plus
        arrayA = PArray.new(101) do |j|
            j
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        arrayC = PArray.new(101) do |j|
            j*3
        end
        
        arrayA = arrayA + arrayB + arrayC

        assert_equal(30300, arrayA.to_a.reduce(:+))
    end

    def test_combine_minus
        arrayA = PArray.new(101) do |j|
            j
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        
        arrayA = arrayA - arrayB

        assert_equal(-5050, arrayA.to_a.reduce(:+))
    end

    def test_combine_mult
        arrayA = PArray.new(101) do |j|
            2
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        
        arrayA = arrayA * arrayB

        assert_equal(20200, arrayA.to_a.reduce(:+))
    end

    def test_combine_div
        arrayA = PArray.new(101) do |j|
            2
        end
        arrayB = PArray.new(101) do |j|
            j*2
        end
        
        arrayA = arrayB / arrayA

        assert_equal(5050, arrayA.to_a.reduce(:+))
    end

    def test_combine_and
        arrayA = PArray.new(101) do |j|
            false
        end
        arrayB = PArray.new(101) do |j|
            true
        end
        
        arrayA = arrayB & arrayA

        assert_equal(false, arrayA.to_a.reduce(:&))
    end

    def test_combine_and2
        arrayA = PArray.new(101) do |j|
            true
        end
        arrayB = PArray.new(101) do |j|
            true
        end
        
        arrayA = arrayB & arrayA

        assert_equal(true, arrayA.to_a.reduce(:&))
    end

    def test_combine_or
        arrayA = PArray.new(101) do |j|
            false
        end
        arrayB = PArray.new(101) do |j|
            true
        end
        
        arrayA = arrayB | arrayA

        assert_equal(true, arrayA.to_a.reduce(:&))
    end

    def test_combine_or2
        arrayA = PArray.new(101) do |j|
            false
        end
        arrayB = PArray.new(101) do |j|
            false
        end
        
        arrayA = arrayB | arrayA

        assert_equal(false, arrayA.to_a.reduce(:&))
    end

    def test_combine_xor
        arrayA = PArray.new(101) do |j|
            true
        end
        arrayB = PArray.new(101) do |j|
            true
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(false, arrayA.to_a.reduce(:&))
    end

    def test_combine_xor2
        arrayA = PArray.new(101) do |j|
            false
        end
        arrayB = PArray.new(101) do |j|
            false
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(false, arrayA.to_a.reduce(:&))
    end

    def test_combine_xor3
        arrayA = PArray.new(101) do |j|
            true
        end
        arrayB = PArray.new(101) do |j|
            false
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(true, arrayA.to_a.reduce(:&))
    end
end
