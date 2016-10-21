require "ikra"
require_relative "unit_test_template"

class RegressionTest < UnitTestCase
    def test_explicit_return
        all_ones = Array.pnew(100) do |j|
            return 1
        end

        assert_equal(100, all_ones.reduce(:+))
    end

    def test_lexical_scope_variable_redefinition
        var = 1

        id_kernel = Array.pnew(100) do |var|
            var
        end

        assert_equal(100, id_kernel.reduce(:+))
    end
end
