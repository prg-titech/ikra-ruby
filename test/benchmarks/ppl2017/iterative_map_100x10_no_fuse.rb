require_relative "../benchmark_base"

class IterativeMapSimple100x10NoFuse < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 12]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_command

            old_data = x

            for r in 0...100
                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
                old_data = y

                y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
                old_data = y

                y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
                old_data = y

                y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
                old_data = y

                y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                old_data.free_memory
                old_data = y
            end

            y
        end
    end
end