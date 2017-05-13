require "ikra"
require_relative "unit_test_template"

class EmptyKernelTest < UnitTestCase
    def test_kernel
        all_zeroes = PArray.new(100) do |j|
            0
        end

        for i in 0..99
            assert_equal(0, all_zeroes[i])
        end
    end
end