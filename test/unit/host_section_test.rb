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
end
