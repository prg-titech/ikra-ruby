require_relative "../benchmark_base"

class IterativeMapSimple100x10NoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = PArray.new(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        y = base

        for r in 0...100
            y = y.map(with_index: true) do |i, indices|
                (i + indices[2]) % 13377
            end

            y = y.map(with_index: true) do |i, indices|
                (i + indices[1]) % 13377
            end

            y = y.map(with_index: true) do |i, indices|
                (i + indices[3]) % 1337
            end

            y = y.map(with_index: true) do |i, indices|
                (i + indices[0]) % 13377
            end

            y = y.map(with_index: true) do |i, indices|
                (i + indices[1]) % 1377
            end
        end

        return y
    end
end