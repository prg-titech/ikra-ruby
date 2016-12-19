require "ikra"
require_relative "unit_test_template"

class StencilTest < UnitTestCase
    def test_stencil_single_parameters
        # CPU computation
        base_array_cpu = Array.new(100) do |j|
            j + 100
        end

        stencil_result_cpu = base_array_cpu.stencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end

        aggregated_cpu = stencil_result_cpu.reduce(:+)


        # GPU computation
        base_array_gpu = Array.pnew(100) do |j|
            j + 100
        end

        stencil_result_gpu = base_array_gpu.pstencil([-1, 0, 1], 10000, use_parameter_array: false) do |p0, p1, p2|
            p0 + p1 + p2
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end

    def test_stencil_array_parameter
        # CPU computation
        base_array_cpu = Array.new(100) do |j|
            j + 100
        end

        stencil_result_cpu = base_array_cpu.stencil([-1, 0, 1], 10000) do |values|
            values[0] + values[1] + values[2]
        end

        aggregated_cpu = stencil_result_cpu.reduce(:+)


        # GPU computation
        base_array_gpu = Array.pnew(100) do |j|
            j + 100
        end

        stencil_result_gpu = base_array_gpu.pstencil([-1, 0, 1], 10000) do |values|
            values[0] + values[1] + values[2]
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end
end
