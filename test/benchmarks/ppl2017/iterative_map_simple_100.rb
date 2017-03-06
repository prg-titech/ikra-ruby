require_relative "../benchmark_base"

class IterativeMapSimple100 < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x
            old_data = x.__call__.to_command

            for r in 0...500
                old_data = y
                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
            end

            y
        end
    end
end