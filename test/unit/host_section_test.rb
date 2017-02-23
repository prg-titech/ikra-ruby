require "ikra"
require_relative "unit_test_template"

class HostSectionTest < UnitTestCase
    def test_simple_host_section
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.pmap do |k|
                k + 1
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_parallel_section_variable
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.pmap do |k|
                k + 1
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_fusion
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 5
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.pmap do |k|
                k + 1
            end.pmap do |k|
                k + 3
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_fusion_and_ssa
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 5
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.pmap do |k|
                k + 1
            end

            a = a.pmap do |k|
                k + 3
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_fusion_and_ssa_2
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 3
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input.pmap do |k|
                k + 1
            end

            a = a.pmap do |k|
                k + 1
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_union_type_return
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 1

            if a == 1
                b = input.pmap do |k|
                    k + 1
                end
            else
                b = input.pmap do |k|
                    k + 10
                end
            end

            b
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_union_type_return_2
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 11
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.pmap do |k|
                    k + 1
                end
            else
                b = input.pmap do |k|
                    k + 10
                end
            end

            b
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_union_type_return_3
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + j + 10 + 100 + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.pmap do |k|
                    k + 1
                end
            else
                # This one is chosen
                b = input.pmap do |k|
                    k + 10
                end
            end

            c = input.pmap do |k|
                k + 100
            end

            b.pzip(c).pmap do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_union_type_return_4
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1 + j + 1 + 10 + 20
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.pmap do |k|
                    k + 1
                end

                c = input.pmap do |k|
                    k + 10
                end
            else
                # This one is chosen
                b = input.pmap do |k|
                    k + 10
                end

                c = input.pmap do |k|
                    k + 20
                end
            end

            b.pzip(c).pmap do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_host_section_with_union_type_return_4_with_ssa
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1 + j + 1 + 10 + 20
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            a = 2

            if a == 1
                b = input.pmap do |k|
                    k + 2
                end

                b = b.pmap do |k|
                    k - 1
                end

                c = input.pmap do |k|
                    k + 5
                end

                c = c.pmap do |k|
                    k + 5
                end
            else
                # This one is chosen
                b = input.pmap do |k|
                    k + 4
                end

                b = b.pmap do |k|
                    k + 6
                end

                c = input.pmap do |k|
                    k + 9
                end

                c = c.pmap do |k|
                    k + 11
                end
            end

            b.pzip(c).pmap do |zipped|
                zipped[0] + zipped[1]
            end
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end

    def test_simple_stencil
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 1
        end

        stencil_result_gpu = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.pstencil([-1, 0, 1], 10000) do |values|
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
        base_array_gpu = Array.pnew(100) do |j|
            j + 100
        end

        stencil_result_gpu = Ikra::Symbolic.host_section(base_array_gpu) do |input|
            # `Ikra::Symbolic.stencil` is computed in the Ruby interpreter
            # TODO: Hoist expression outside of host_section?
            input.pstencil(Ikra::Symbolic.stencil(directions: 1, distance: 1), 10000) do |values|
                values[-1] + values[0] + values[1]
            end
        end 

        aggregated_gpu = stencil_result_gpu.reduce(:+)


        # Compare results
        assert_equal(aggregated_cpu, aggregated_gpu)
    end

    def test_iterative_map_update
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 10
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|

            a = input

            for i in 1...10
                a = a.pmap do |k|
                    k + 1
                end
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end
end
