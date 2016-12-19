require "ikra"
require_relative "unit_test_template"

class ZipTest < UnitTestCase
    def test_simple_zip
        array_1 = Array.pnew(100) do |j|
            j
        end

        array_2 = Array.pnew(100) do |j|
            j * j
        end

        result = array_1.pzip(array_2).pmap do |zipped|
            zipped[0] + zipped[1]
        end

        assert_equal(333300, result.reduce(:+))
    end

    def test_simple_zip_3
        # GPU  
        array_1 = Array.pnew(100) do |j|
            j
        end

        array_2 = Array.pnew(100) do |j|
            j * j
        end

        array_3 = Array.pnew(100) do |j|
            j * j * j
        end

        result_gpu = array_1.pzip(array_2, array_3).pmap do |zipped|
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


        assert_equal(result_cpu.reduce(:+), result_gpu.reduce(:+))
    end

    def test_zip_indirect_access
        array_1 = Array.pnew(100) do |j|
            j
        end

        array_2 = Array.pnew(100) do |j|
            j * j
        end

        array_3 = Array.pnew(100) do |j|
            j % 2 == 0 ? 0 : 1
        end

        result = array_1.pzip(array_2, array_3).pmap do |zipped|
            zipped[zipped[2]]
        end

        assert_equal(169100, result.reduce(:+))
    end

    def test_return_zip_struct
        array_1 = Array.pnew(100) do |j|
            j * j
        end

        array_2 = Array.pnew(100) do |j|
            j + 0.1
        end

        result = array_1.pzip(array_2).pmap do |v|
            v
        end

        # Reduce on CPU
        expected_value = 0
        result.each do |zipped|
            expected_value = expected_value + zipped[0] + zipped[1] * 100
        end

        assert_in_delta(824350.0, expected_value, 0.01)
    end
end
