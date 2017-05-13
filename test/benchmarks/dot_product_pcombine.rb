require_relative "benchmark_base"

class DotProductCombine < Test::Unit::TestCase
    include BenchmarkBase

    SIZE = 30_000_000

    def execute
        arr1 = PArray.new(SIZE, block_size: 1024) do |index| index % 25000 end
        arr2 = PArray.new(SIZE, block_size: 1024) do |index| (index + 101) % 25000 end

        return arr1.combine(arr2, block_size: 1024) do |p1, p2|
            p1 * p2
        end
    end
end