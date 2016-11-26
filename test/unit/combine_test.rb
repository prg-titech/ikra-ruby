require "ikra"
require_relative "unit_test_template"

class CombineTest < UnitTestCase
    def test_combine
        arrayA = Array.pnew(101) do |j|
            j
        end

        arrayB = Array.pnew(101) do |j|
            5
        end

        arrayC = Array.pnew(101) do |j|
            j * 2
        end
        
        arrayA = arrayA.pcombine(arrayB, arrayC) do |a,b,c|
            a + b + c
        end

        assert_equal(15655, arrayA.reduce(:+))
    end


    def test_combine_no_arg
        arrayA = Array.pnew(101) do |j|
            j
        end
        
        arrayA = arrayA.pcombine() do |a|
            a
        end

        assert_equal(5050, arrayA.reduce(:+))
    end

    def test_combine_three_arg
        arrayA = Array.pnew(101) do |j|
            j
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        arrayC = Array.pnew(101) do |j|
            j*3
        end
        arrayD = Array.pnew(101) do |j|
            j*4
        end
        
        arrayA = arrayA.pcombine(arrayB, arrayC, arrayD) do |a,b,c,d|
            a+b+c+d
        end

        assert_equal(50500, arrayA.reduce(:+))
    end

    def test_combine_plus
        arrayA = Array.pnew(101) do |j|
            j
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        
        arrayA = arrayA + arrayB

        assert_equal(15150, arrayA.reduce(:+))
    end

    def test_combine_plus_plus
        arrayA = Array.pnew(101) do |j|
            j
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        arrayC = Array.pnew(101) do |j|
            j*3
        end
        
        arrayA = arrayA + arrayB + arrayC

        assert_equal(30300, arrayA.reduce(:+))
    end

    def test_combine_minus
        arrayA = Array.pnew(101) do |j|
            j
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        
        arrayA = arrayA - arrayB

        assert_equal(-5050, arrayA.reduce(:+))
    end

    def test_combine_mult
        arrayA = Array.pnew(101) do |j|
            2
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        
        arrayA = arrayA * arrayB

        assert_equal(20200, arrayA.reduce(:+))
    end

    def test_combine_div
        arrayA = Array.pnew(101) do |j|
            2
        end
        arrayB = Array.pnew(101) do |j|
            j*2
        end
        
        arrayA = arrayB / arrayA

        assert_equal(5050, arrayA.reduce(:+))
    end

    def test_combine_and
        arrayA = Array.pnew(101) do |j|
            false
        end
        arrayB = Array.pnew(101) do |j|
            true
        end
        
        arrayA = arrayB & arrayA

        assert_equal(false, arrayA.reduce(:&))
    end

    def test_combine_and2
        arrayA = Array.pnew(101) do |j|
            true
        end
        arrayB = Array.pnew(101) do |j|
            true
        end
        
        arrayA = arrayB & arrayA

        assert_equal(true, arrayA.reduce(:&))
    end

    def test_combine_or
        arrayA = Array.pnew(101) do |j|
            false
        end
        arrayB = Array.pnew(101) do |j|
            true
        end
        
        arrayA = arrayB | arrayA

        assert_equal(true, arrayA.reduce(:&))
    end

    def test_combine_or2
        arrayA = Array.pnew(101) do |j|
            false
        end
        arrayB = Array.pnew(101) do |j|
            false
        end
        
        arrayA = arrayB | arrayA

        assert_equal(false, arrayA.reduce(:&))
    end

    def test_combine_xor
        arrayA = Array.pnew(101) do |j|
            true
        end
        arrayB = Array.pnew(101) do |j|
            true
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(false, arrayA.reduce(:&))
    end

    def test_combine_xor2
        arrayA = Array.pnew(101) do |j|
            false
        end
        arrayB = Array.pnew(101) do |j|
            false
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(false, arrayA.reduce(:&))
    end

    def test_combine_xor3
        arrayA = Array.pnew(101) do |j|
            true
        end
        arrayB = Array.pnew(101) do |j|
            false
        end
        
        arrayA = arrayB ^ arrayA

        assert_equal(true, arrayA.reduce(:&))
    end
end
