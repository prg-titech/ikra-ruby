require "test/unit"
require "ikra"

class SimpleFloatKernelTest < Test::Unit::TestCase
    def test_kernel
        Ikra::Configuration.codegen_expect_file_name = nil

        all_floats = Array.pnew(100) do |j|
            1.12
        end

        for i in 0..99
            assert_in_delta(1.12, all_floats[i], 0.001)
        end
    end
end