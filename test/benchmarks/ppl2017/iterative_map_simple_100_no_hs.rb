require_relative "../benchmark_base"

class IterativeMapSimple100NoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = PArray.new(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        for r in 0...500
            base = base.map(with_index: true) do |i, indices|
                (i + indices[2]) % 13377
            end
        end

        return base
    end
end