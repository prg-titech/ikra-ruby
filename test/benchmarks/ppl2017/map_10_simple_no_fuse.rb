require_relative "../benchmark_base"

class Map10SimpleNoFuse < Test::Unit::TestCase
    include BenchmarkBase

    def execute
        base = Array.pnew(dimensions: [20, 500, 500, 12]) do |indices|
            (7 + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
        end

        return Ikra::Symbolic.host_section(base) do |base|
            y = base.__call__.to_command
            old_data = base

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y = y.__call__.to_command.pmap(with_index: true) do |i, indices|
                (i + indices[0] + indices[1] + indices[2] + indices[3]) % 1337
            end

            old_data.free_memory
            old_data = y

            y
        end
    end
end