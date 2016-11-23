require "ikra"
require_relative "unit_test_template"

class StencilTest < UnitTestCase
    def test_stencil
        # CPU computation
        base_array_cpu = Array.new(100) do |j|
            j + 100
        end

        stencil_result_cpu = base_array_cpu.pstencil([-1, 0, 1], 10000) do |p0, p1, p2|
            p0 + p1 + p2
        end

        aggregated_cpu = stencil_result_cpu.reduce(:+)


        # GPU computation
        base_array_gpu = Array.pnew(100) do |j|
            j + 100
        end

        stencil_result_gpu = base_array_gpu.pstencil([-1, 0, 1], 10000) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_cpu)
    end
end
