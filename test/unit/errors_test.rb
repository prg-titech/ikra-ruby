require "ikra"
require_relative "unit_test_template"

class ErrorsTest < UnitTestCase
    def test_top_level_method_not_found
        array = Array.pnew(100) do |j|
            this_method_does_not_exist(1, 2)
        end

        assert_raise NameError do
            array.execute
        end
    end

    def test_primitive_instance_method_not_found
        array = Array.pnew(100) do |j|
            5.method_does_not_exist
        end

        assert_raise NotImplementedError do
            array.execute
        end
    end
end