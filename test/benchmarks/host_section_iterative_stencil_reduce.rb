require_relative "benchmark_base"

class HostSectionIterativeStencil < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        array_gpu = PArray.new(90210) do |j|
            j % 2
        end

        # GPU calculation
        return Ikra::Symbolic.host_section(array_gpu) do |input|
            a = input

            while (a.reduce do |a, b| a + b end).__call__.__to_host_array__[0] < 10000000
                a = a.stencil([-1, 0, 1], 1) do |values|
                    (values[-1] - values[0] - values[1] + 7)
                end
            end

            a
        end
    end

    def expected
        array_cpu = Array.new(90210) do |j|
            j % 2
        end

        num_iter = 0    # Count iterations
        return begin
            a = array_cpu

            while (a.reduce do |a, b| a + b end) < 10000000
                num_iter = num_iter + 1
                a = a.stencil([-1, 0, 1], 1, use_parameter_array: false) do |p0, p1, p2|
                    (p0 - p1 - p2 + 7)
                end
            end

            a
        end
    end
end
