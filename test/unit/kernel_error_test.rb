require "ikra"
require "ikra"
require_relative "unit_test_template"

class KernelErrorTest < UnitTestCase
    def test_memory_access_violation
        lex_array = (0...100).to_a

        array = Array.pnew(100) do |j|
            # TODO: This should actually wrap around or return nil
            lex_array[j - 1000000000] + 1
        end

        assert_raise Ikra::Errors::CudaErrorIllegalAddress do
            array.execute
        end
    end

    def test_invalid_operand_type
        array = Array.pnew(100) do |index|
            value = 1

            if index % 2 == 1
                value = true
            end

            1 + value
        end

        array.execute
    end
end