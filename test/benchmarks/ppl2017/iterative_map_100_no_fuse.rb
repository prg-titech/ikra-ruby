require_relative "../benchmark_base"

class IterativeMap100NoFuse < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = Array.pnew(dimensions: [20, 500, 500, 2]) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_command
            old_data = y

            for r in 0...500
                old_data = y
                y = y.pmap(with_index: true) do |i, indices|
                    ((i % 938) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
                end

                old_data.free_memory
            end

            y
        end
    end
end