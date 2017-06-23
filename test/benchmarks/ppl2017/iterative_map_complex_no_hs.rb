require_relative "../benchmark_base"

class IterativeMapComplexNoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 12]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        y = base

        for r in 0...200
            if r % 2 == 0
                if r % 3 == 0
                    y = y.pmap(with_index: true) do |i, indices|
                        (i + indices[3]) % 77689
                    end
                else
                    y = y.pmap(with_index: true) do |i, indices|
                        (i + indices[0]) % 11799
                    end
                end
            else
                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 1337
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 8888888
                end
            end

            y = y.pmap(with_index: true) do |i, indices|
                (i + indices[2]) % 6678
            end
        end

        return y
    end
end