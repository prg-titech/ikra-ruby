require_relative "benchmark_base"

class DotProductZipped < Test::Unit::TestCase
    include BenchmarkBase

    SIZE = 25000

    def execute
        arr1 = Array.pnew(SIZE) do |index| index end
        arr2 = Array.pnew(SIZE) do |index| index + 100 end

        return arr1.pzip(arr2).pmap do |zipped|
            zipped[0] * zipped[1]
        end
    end

    def expected
        arr1 = Array.new(SIZE) do |index| index end
        arr2 = Array.new(SIZE) do |index| index + 100 end

        return arr1.zip(arr2).map do |zipped|
            zipped[0] * zipped[1]
        end
    end
end