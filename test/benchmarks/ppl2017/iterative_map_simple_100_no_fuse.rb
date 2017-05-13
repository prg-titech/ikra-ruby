require_relative "../benchmark_base"

class IterativeMapSimple100NoFuse < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = PArray.new(dimensions: [20, 500, 500, 2]) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_pa
            old_data = y

            for r in 0...500
                old_data = y
                y = y.map(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
            end

            y
        end
    end
end