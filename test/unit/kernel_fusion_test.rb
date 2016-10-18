require "ikra"
require_relative "unit_test_template"

class KernelFusionTest < UnitTestCase
    def test_fusion_2
        base_array = Array.pnew(100) do |j|
            j + 1
        end

        mapped_array = base_array.pmap do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1), mapped_array[i])
        end
    end

    def test_fusion_3
        base_array = Array.pnew(100) do |j|
            j + 1
        end

        mapped_array = base_array.pmap do |j|
            j * j
        end

        mapped_array = mapped_array.pmap do |j|
            j * j
        end

        for i in 0..99
            assert_equal((i + 1) * (i + 1) * (i + 1) * (i + 1), mapped_array[i])
        end
    end

end