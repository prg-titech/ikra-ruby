require "ikra"
require_relative "unit_test_template"

class DebugTest < UnitTestCase
    def test_stencil_host_section
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.pstencil([-1, 0, 1], 10000) do |values|
                values[-1] + values[0] + values[1]
            end 
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end
end