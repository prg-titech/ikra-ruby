require_relative "../benchmark_base"

class Stencil10NoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = PArray.new(dimensions: [20, 500, 500, 2]) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        base = base.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
        end

        return base
    end
end