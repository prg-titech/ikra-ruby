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
            values[-1] + values[0] + values[1]
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end

    def test_2d_stencil
        # GPU computation
        base_array_gpu = Array.pnew(dimensions: [7, 5]) do |index|
            10 * index[0] + index[1]
        end

        stencil_result_gpu = base_array_gpu.pstencil([[-1, -1], [0, -1], [0, 1], [0, 0]], 10000) do |values|
            values[-1][-1] + values[0][-1] + values[0][1] + values[0][0]
        end

        # Compare results
        assert_equal([10000, 10000, 10000, 10000, 10000, 10000, 33, 37, 41, 10000, 10000, 73, 77, 81, 10000, 10000, 113, 117, 121, 10000, 10000, 153, 157, 161, 10000, 10000, 193, 197, 201, 10000, 10000, 233, 237, 241, 10000], stencil_result_gpu.to_a)
    end

    def test_non_constant_stencil
        # CPU computation
        base_array_cpu = Array.new(100) do |j|
            j + 100
        end

        stencil_result_cpu = base_array_cpu.stencil([-2, -1, 0, 1, 2], 10000) do |values|
            if values[2] % 2 == 0
                x = 1
            else
                x = -1
            end

            values[2*x+2] + values[2]
        end

        aggregated_cpu = stencil_result_cpu.reduce(:+)


        # GPU computation
        base_array_gpu = Array.pnew(100) do |j|
            j + 100
        end

        stencil_result_gpu = base_array_gpu.pstencil([-2, -1, 0, 1, 2], 10000) do |values|
            if values[0] % 2 == 0
                x = 1
            else
                x = -1
            end

            values[2*x] + values[0]
        end

        aggregated_gpu = stencil_result_gpu.reduce(:+)



        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end
end
