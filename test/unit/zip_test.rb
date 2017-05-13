require "ikra"
require_relative "unit_test_template"

class ZipTest < UnitTestCase
    def test_simple_zip
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        result = array_1.zip(array_2).map do |zipped|
            zipped[0] + zipped[1]
        end

        assert_equal(333300, result.to_a.reduce(:+))
    end

    def test_reuse_zip_type
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        array_2b = PArray.new(100) do |j|
            j * j
        end

        array_3 = PArray.new(100) do |j|
            j * j * j
        end

        result_1 = array_1.zip(array_2).map do |zipped|
            zipped
        end

        result_2 = array_2b.zip(array_3).map do |zipped|
            zipped
        end

        result = result_1.combine(result_2) do |z1, z2|
            z1[0] + z1[1] + z2[0] + z2[1]
        end

        # Cannot really test this here, but there should be only one struct definition
        # in the source code for <int, int>
        assert_equal(25164150, result.to_a.reduce(:+))
    end

    def test_simple_zip_3
        # GPU  
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        array_3 = PArray.new(100) do |j|
            j * j * j
        end

        result_gpu = array_1.zip(array_2, array_3).map do |zipped|
            zipped[0] + zipped[1] + zipped[2]
        end


        # CPU  
        array_1_c = Array.new(100) do |j|
            j
        end

        array_2_c = Array.new(100) do |j|
            j * j
        end

        array_3_c = Array.new(100) do |j|
            j * j * j
        end

        result_cpu = array_1_c.zip(array_2_c, array_3_c).map do |zipped|
            zipped[0] + zipped[1] + zipped[2]
        end


        assert_equal(result_cpu.reduce(:+), result_gpu.to_a.reduce(:+))
    end

    def test_nested_zip
        # GPU  
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        array_3 = PArray.new(100) do |j|
            j * j * j
        end

        result_gpu = array_1.zip(array_2).zip(array_3).map do |zipped|
            zipped[0][0] + zipped[0][1] + zipped[1]
        end


        # CPU  
        array_1_c = Array.new(100) do |j|
            j
        end

        array_2_c = Array.new(100) do |j|
            j * j
        end

        array_3_c = Array.new(100) do |j|
            j * j * j
        end

        result_cpu = array_1_c.zip(array_2_c).zip(array_3_c).map do |zipped|
            zipped[0][0] + zipped[0][1] + zipped[1]
        end


        assert_equal(result_cpu.reduce(:+), result_gpu.to_a.reduce(:+))
    end

    def test_zip_indirect_access
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        array_3 = PArray.new(100) do |j|
            j % 2 == 0 ? 0 : 1
        end

        result = array_1.zip(array_2, array_3).map do |zipped|
            zipped[zipped[2]]
        end

        assert_equal(169100, result.to_a.reduce(:+))
    end

    def test_return_zip_struct
        array_1 = PArray.new(100) do |j|
            j * j
        end

        array_2 = PArray.new(100) do |j|
            j + 0.1
        end

        result = array_1.zip(array_2).map do |v|
            v
        end

        # Reduce on CPU
        expected_value = 0
        result.each do |zipped|
            expected_value = expected_value + zipped[0] + zipped[1] * 100
        end

        assert_in_delta(824350.0, expected_value, 0.01)
    end

    def test_return_nested_zip
        # GPU  
        array_1 = PArray.new(100) do |j|
            j
        end

        array_2 = PArray.new(100) do |j|
            j * j
        end

        array_3 = PArray.new(100) do |j|
            j * j * j
        end

        result_gpu = array_1.zip(array_2).zip(array_3).map do |zipped|
            zipped
        end

        # Reduce on CPU
        result_reduced = result_gpu.to_a.reduce(0) do |acc, n|
            acc + n[0][0] + n[0][1] + n[1]
        end

        # CPU  
        array_1_c = Array.new(100) do |j|
            j
        end

        array_2_c = Array.new(100) do |j|
            j * j
        end

        array_3_c = Array.new(100) do |j|
            j * j * j
        end

        result_cpu = array_1_c.zip(array_2_c).zip(array_3_c).map do |zipped|
            zipped[0][0] + zipped[0][1] + zipped[1]
        end


        assert_equal(result_cpu.reduce(:+), result_reduced)
    end
end
