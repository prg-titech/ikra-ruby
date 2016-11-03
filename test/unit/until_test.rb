require "ikra"
require_relative "unit_test_template"

class UntilTest < UnitTestCase
    def test_while
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            while x < 10
                x = x + 1
                y = y + x
            end
            y
        end

        assert_in_delta(5500, array.reduce(:+), 1)
    end

    def test_until1
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            until x == 10
                x = x + 1
                y = y + x
            end
            y
        end

        assert_in_delta(5500, array.reduce(:+), 1)
    end

    def test_until2
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            until x == -1
                x = x - 1
                y = y + x
            end
            y
        end

        assert_in_delta(-100, array.reduce(:+), 1)
    end

    def test_while_post
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            begin
                x = x + 1
                y = y + x
            end while x < 10
            y
        end

        assert_in_delta(5500, array.reduce(:+), 1)
    end

    def test_until_post
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            begin
                x = x + 1
                y = y + x
            end until x >= 10
            y
        end

        assert_in_delta(5500, array.reduce(:+), 1)
    end
end
