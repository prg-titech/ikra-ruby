require_relative "../benchmark_base"

class Map10NoFuse < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = PArray.new(dimensions: [20, 500, 500, 2]) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        return Ikra::Symbolic.host_section(base) do |base|

            base = base.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 938) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 939) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 940) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 941) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 942) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 943) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 944) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 945) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 946) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end.__call__.to_pa

            base = base.map(with_index: true) do |i, indices|
                ((i % 947) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end

            base
        end
    end
end