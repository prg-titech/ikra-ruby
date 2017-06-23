require_relative "../benchmark_base"

class Map10SimpleNoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = Array.pnew(dimensions: [20, 500, 500, 12]) do |indices|
            (7 + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        base = base.pmap(with_index: true) do |i, indices|
            (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        return base
    end
end