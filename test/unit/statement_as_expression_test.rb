require "ikra"
require_relative "unit_test_template"

class StatementAsExpressionTest < UnitTestCase
    def test_if_assignment
        array = Array.pnew(100) do |j|
            x = 0
            x = if j % 2 == 0; x + 1 else 1 end
            x + 1
        end

        assert_equal(100 + 100, array.reduce(:+))
    end

    def test_nested_begin
        array = Array.pnew(100) do |j|
            x = 0
            x = begin 
                    begin 
                        begin
                            2
                        end
                    end 
                end
            x + 1
        end

        assert_equal(3 * 100, array.reduce(:+))
    end

    def test_begin_with_statements
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            x = begin 
                    y = 4
                    2
                end
            x + y + 1       # 2 + 4 + 1 = 7
        end

        assert_equal(7 * 100, array.reduce(:+))
    end

    def test_nested_begin_with_statements
        array = Array.pnew(100) do |j|
            x = 0
            y = 0
            x = begin 
                    y = begin
                        4
                    end

                    2
                end
            x + y + 1       # 2 + 4 + 1 = 7
        end

        assert_equal(7 * 100, array.reduce(:+))
    end
end