require_relative "benchmark_base"

class SquareBenchmark < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        array = (1..10000).to_a
        return array.to_pa.map do |value|
            value * value
        end
    end

    def expected
        return (1..10000).to_a.map do |value|
            value * value
        end
    end
end