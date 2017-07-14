require_relative "benchmark_base"

class DotProductZipped < Test::Unit::TestCase
    include BenchmarkBase

    SIZE = 30_000_000

    def execute
        arr1 = PArray.new(SIZE, block_size: 1024) do |index| index % 25000 end
        arr2 = PArray.new(SIZE, block_size: 1024) do |index| (index + 101) % 25000 end

        return arr1.zip(arr2).map(block_size: 1024) do |zipped|
            zipped[0] * zipped[1]
        end
    end
end