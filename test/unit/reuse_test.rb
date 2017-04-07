require "ikra"
require_relative "unit_test_template"

class ReuseTest < UnitTestCase
    def test_keep_and_create_array_id_command
        array = Array.pnew(100) do |i|
            i + 1
        end

        array2 = array.pmap(keep: true) do |j|
            j + 1
        end

        # Launch array2
        array2.to_a

        # Get pointer to location in global memory
        array2_base = array2.read_from_memory

        array3 = array2_base.pmap do |k|
            k + 5
        end

        # Compute expected value
        expected = Array.new(100) do |i|
            i + 1 + 1 + 5
        end

        assert_equal(expected, array3.to_a)
    end

    def test_binary_cache
        array = Array.pnew([20, 500, 500, 2].reduce(:*), keep: true) do |i2|
            i2 + 19
        end

        # Launch array
        array.to_a
        array = array.read_from_memory

        for x in 0...500
            puts "ITERATION: #{x}"

            array = array.pmap(keep: true) do |j2|
                j2 + 2
            end

            array.to_a
            array = array.read_from_memory
        end

        array = array.pmap do |j3|
            j3 + 1
        end

        # Compute expected value
        expected = Array.new(10100) do |i|
            i + 19 + 2 * 200 + 1
        end

        assert_equal(expected, array.to_a)
    end
end
