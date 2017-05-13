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

def another_method
    return 456
end

class StaticMethodTest < UnitTestCase
    def test_static_method
        base_array = PArray.new(100) do |j|
            DummyClass.some_method + 1
        end

        assert_equal(124 * 100, base_array.to_a.reduce(:+))
    end

    def test_top_level_method_with_explicit_receiver
        base_array = PArray.new(100) do |j|
            Object.another_method + 1
        end

        assert_equal(457 * 100, base_array.to_a.reduce(:+))
    end

    def test_top_level_method_with_implicit_receiver
        base_array = PArray.new(100) do |j|
            another_method + 1
        end

        assert_equal(457 * 100, base_array.to_a.reduce(:+))
    end
end