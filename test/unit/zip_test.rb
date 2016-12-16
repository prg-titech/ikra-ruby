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

        assert_in_delta(333300, result.reduce(:+), 1)
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


        assert_in_delta(result_cpu.reduce(:+), result_gpu.reduce(:+), 1)
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

        assert_in_delta(169100, result.reduce(:+), 1)
    end
end
