require_relative "benchmark_base"

class HostSectionIterativeMap < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        return Ikra::Symbolic.host_section(array_gpu) do |input|

            a = input

            for i in 1...100000
                a = a.pmap do |k|
                    k + 1
                end
            end

            a
        end
    end

    def expected
        return Array.new(511) do |j|
            j + 100000
        end
    end
end