require_relative "../benchmark_base"

class IterativeStencil100NoFuse < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = PArray.new(dimensions: DIMS) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_pa
            old_data = x
            old_old_data = x

            for r in 0...200
                old_old_data = old_data
                old_data = y

                y = y.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
                ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
                end

                if r > 1
                    old_old_data.free_memory
                end
            end

            y
        end
    end
end