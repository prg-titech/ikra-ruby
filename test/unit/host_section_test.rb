require "ikra"
require_relative "unit_test_template"

class HostSectionTest < UnitTestCase
    def test_simple_host_section
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.map do |k|
                k + 1
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_parallel_section_variable
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.map do |k|
                k + 1
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_fusion
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 5
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.map do |k|
                k + 1
            end.map do |k|
                k + 3
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_fusion_and_ssa
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 5
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.map do |k|
                k + 1
            end

            a = a.map do |k|
                k + 3
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_fusion_and_ssa_2
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 3
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.map do |k|
                k + 1
            end

            a = a.map do |k|
                k + 1
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_union_type_return
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 1

            if a == 1
                b = input.map do |k|
                    k + 1
                end
            else
                b = input.map do |k|
                    k + 10
                end
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_union_type_return_2
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 11
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.map do |k|
                    k + 1
                end
            else
                b = input.map do |k|
                    k + 10
                end
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_union_type_return_3
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + j + 10 + 100 + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.map do |k|
                    k + 1
                end
            else
                # This one is chosen
                b = input.map do |k|
                    k + 10
                end
            end

            c = input.map do |k|
                k + 100
            end

            b.zip(c).map do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_union_type_return_4
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1 + j + 1 + 10 + 20
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.map do |k|
                    k + 1
                end

                c = input.map do |k|
                    k + 10
                end
            else
                # This one is chosen
                b = input.map do |k|
                    k + 10
                end

                c = input.map do |k|
                    k + 20
                end
            end

            b.zip(c).map do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_host_section_with_union_type_return_4_with_ssa
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1 + j + 1 + 10 + 20
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.map do |k|
                    k + 2
                end

                b = b.map do |k|
                    k - 1
                end

                c = input.map do |k|
                    k + 5
                end

                c = c.map do |k|
                    k + 5
                end
            else
                # This one is chosen
                b = input.map do |k|
                    k + 4
                end

                b = b.map do |k|
                    k + 6
                end

                c = input.map do |k|
                    k + 9
                end

                c = c.map do |k|
                    k + 11
                end
            end

            b.zip(c).map do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_simple_stencil
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1
        end

        stencil_result_gpu = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.stencil([-1, 0, 1], 10000) do |values|
                values[-1] + values[0] + values[1]
            end 
        end

        stencil_result_cpu = array_cpu.stencil([-1, 0, 1], 10000) do |values|
            values[0] + values[1] + values[2]
        end

        assert_equal(stencil_result_cpu , stencil_result_gpu.to_a)
    end

    def test_generated_stencil_with_interpreter_method
        # CPU computation
        base_array_cpu = Array.new(100) do |j|
            j + 100
        end

        stencil_result_cpu = base_array_cpu.stencil([-1, 0, 1], 10000) do |values|
            values[0] + values[1] + values[2]
        end

        aggregated_cpu = stencil_result_cpu.reduce(:+)


        # GPU computation
        base_array_gpu = PArray.new(100) do |j|
            j + 100
        end

        stencil_result_gpu = Ikra::Symbolic.host_section(base_array_gpu) do |input|
            # `Ikra::Symbolic.stencil` is computed in the Ruby interpreter
            # TODO: Hoist expression outside of host_section?
            input.stencil(Ikra::Symbolic.stencil(directions: 1, distance: 1), 10000) do |values|
                values[-1] + values[0] + values[1]
            end
        end 

        aggregated_gpu = stencil_result_gpu.to_a.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end

    def test_iterative_map_update
        array_gpu = PArray.new(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 10
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|

            a = input

            for i in 1...10
                a = a.map do |k|
                    k + 1
                end
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_iterative_stencil_update
        array_gpu = PArray.new(1195) do |j|
            j + 1
        end

        array_cpu = Array.new(1195) do |j|
            j + 1
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            for i in 1...10
                a = a.stencil([-1, 0, 1], 5) do |values|
                    values[-1] - values[0] + values[1] + 1
                end
            end

            a
        end

        for i in 1...10
            array_cpu = array_cpu.stencil([-1, 0, 1], 5, use_parameter_array: false) do |p0, p1, p2|
                p0 - p1 + p2 + 1
            end
        end

        assert_equal(array_cpu.to_a , section_result.to_a)
    end

    def test_iterative_map_update_with_reduce_criteria
        array_gpu = PArray.new(102900) do |j|
            j % 7
        end

        array_cpu = Array.new(102900) do |j|
            j % 7
        end

        # GPU calculation
        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            while (a.reduce do |a, b| a + b end).__call__.__to_host_array__[0] < 10000000
                a = a.map do |k|
                    k + 1
                end
            end

            a
        end

        # CPU calculation
        num_iter = 0    # Count iterations
        result = begin
            a = array_cpu

            while (a.reduce do |a, b| a + b end) < 10000000
                num_iter = num_iter + 1
                a = a.map do |k|
                    k + 1
                end
            end

            a
        end

        assert_equal(result.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_iterative_stencil_update_with_reduce_criteria
        array_gpu = PArray.new(902) do |j|
            j % 2
        end

        array_cpu = Array.new(902) do |j|
            j % 2
        end

        # GPU calculation
        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            while (a.reduce do |a, b| a + b end).__call__.__to_host_array__[0] < 10000
                a = a.stencil([-1, 0, 1], 1) do |values|
                    (values[-1] - values[0] - values[1] + 7)
                end
            end

            a
        end

        # CPU calculation
        num_iter = 0    # Count iterations
        result = begin
            a = array_cpu

            while (a.reduce do |a, b| a + b end) < 10000
                num_iter = num_iter + 1
                a = a.stencil([-1, 0, 1], 1, use_parameter_array: false) do |p0, p1, p2|
                    (p0 - p1 - p2 + 7)
                end
            end

            a
        end

        assert_equal(result.reduce(:+) , section_result.to_a.reduce(:+))
    end

    def test_iterative_stencil_update_and_index
        array_gpu = PArray.new(1195) do |j|
            j + 1
        end

        array_cpu = Array.new(1195) do |j|
            j + 1
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            for i in 1...10
                a = a.stencil([-1, 0, 1], 5, with_index: true) do |values, index|
                    values[-1] - values[0] + values[1] + 1 + (index % 11)
                end
            end

            a
        end

        for i in 1...10
            array_cpu = array_cpu.stencil([-1, 0, 1], 5, use_parameter_array: false, with_index: true) do |p0, p1, p2, index|
                p0 - p1 + p2 + 1 + (index % 11)
            end
        end

        assert_equal(array_cpu.to_a , section_result.to_a)
    end

    def test_map_with_2d_index
        array_gpu = PArray.new(dimensions: [2, 3]) do |index|
            index[0] * 10 + index[1]
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            a.map(with_index: true) do |value, index|
                value + 1000 * index[0] + 100 * index[1]
            end
        end

        assert_equal([0, 101, 202, 1010, 1111, 1212] , section_result.to_a)
    end

    def test_iterative_map_with_2d_index
        array_gpu = PArray.new(dimensions: [2, 3]) do |index|
            index[0] * 10 + index[1]
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            for i in 1..5
                a = a.map(with_index: true) do |value, index|
                    value + 1000 * index[0] + 100 * index[1]
                end
            end

            a
        end

        assert_equal([0, 501, 1002, 5010, 5511, 6012], section_result.to_a)
    end

    def test_iterative_stencil_update_2d_and_index
        base_array_gpu = PArray.new(dimensions: [7, 5]) do |index|
            10 * index[0] + index[1]
        end

        # Host section result
        section_result = Ikra::Symbolic.host_section(base_array_gpu) do |input|
            a = input

            for i in 0...3
                a = a.stencil([[-1, -1], [0, -1], [0, 1], [0, 0]], 10000, with_index: true) do |values, indices|
                    values[-1][-1] + values[0][-1] + values[0][1] + values[0][0] + indices[1] - indices[0]
                end
            end

            a
        end

        # Expected result
        b = base_array_gpu
        for i in 0...3
            b = b.stencil([[-1, -1], [0, -1], [0, 1], [0, 0]], 10000, with_index: true) do |values, indices|
                values[-1][-1] + values[0][-1] + values[0][1] + values[0][0] + indices[1] - indices[0]
            end
        end

        # Compare results
        assert_equal(b.to_a, section_result.to_a)
    end

    def test_iterative_stencil_update_3d_and_index
        base_array_gpu = PArray.new(dimensions: [7, 5, 3]) do |index|
            10 * index[0] + index[1] - index[2]
        end

        # Host section result
        section_result = Ikra::Symbolic.host_section(base_array_gpu) do |input|
            a = input

            for i in 0...3
                a = a.stencil([[-1, -1, 1], [0, -1, -1], [0, 1, 0], [0, 0, 0]], 10000, with_index: true) do |values, indices|
                    values[-1][-1][1] + values[0][-1][-1] + values[0][1][0] + values[0][0][0] + indices[1] - indices[0]
                end
            end

            a
        end

        # Expected result
        b = base_array_gpu
        for i in 0...3
            b = b.stencil([[-1, -1, 1], [0, -1, -1], [0, 1, 0], [0, 0, 0]], 10000, with_index: true) do |values, indices|
                values[-1][-1][1] + values[0][-1][-1] + values[0][1][0] + values[0][0][0] + indices[1] - indices[0]
            end
        end

        # Compare results
        assert_equal(b.to_a, section_result.to_a)
    end
end
