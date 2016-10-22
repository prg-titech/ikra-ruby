require "ikra"
require_relative "unit_test_template"

def top_level_method
    return 123
end

class DummyClass
    class << self
        def some_method
            return 123
        end
    end
end

class TopLevelMethodTest < UnitTestCase
    def test_kernel
        base_array = Array.pnew(100) do |j|
            DummyClass.some_method + 1
        end

        assert_equal(124 * 100, base_array.reduce(:+))
    end
end