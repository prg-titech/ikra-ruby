require_relative "../benchmark_base"

class IterativeMapComplexNoFuse < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_command
            old_data = x

            for r in 0...200
                if r % 2 == 0
                    if r % 3 == 0
                        y = y.pmap(with_index: true) do |i, indices|
                            (i + indices[3]) % 77689
                        end

                        old_data.free_memory
                        old_data = y
                    else
                        y = y.pmap(with_index: true) do |i, indices|
                            (i + indices[0]) % 11799
                        end

                        old_data.free_memory
                        old_data = y
                    end
                else
                    y = y.pmap(with_index: true) do |i, indices|
                        (i + indices[2]) % 1337
                    end

                    old_data.free_memory
                    old_data = y

                    y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                        (i + indices[2]) % 8888888
                    end

                    old_data.free_memory
                    old_data = y
                end

                y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 6678
                end

                old_data.free_memory
                old_data = y
            end

            y
        end
    end
end