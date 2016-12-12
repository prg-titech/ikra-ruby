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
end
