require_relative "../benchmark_base"

class IterativeMapSimple100x10 < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_command
            old_data = x

            for r in 0...100
                old_data = y

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 13377
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[1]) % 13377
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[3]) % 1337
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[0]) % 13377
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[1]) % 1377
                end

                old_data.free_memory
            end

            y
        end
    end


    # --- VERIFICATION CODE BELOW ---

    # Verify once then deactivate. Takes too long...
    def expected_deactivated
        base = Array.new(DIMS.reduce(:*)) do |idx|
            indices = build_indices(idx)
            (indices[2]) % 133777
        end

        y = base

        for i in 0...200
            y = y.map.with_index do |i, idx|
                indices = build_indices(idx)
                (i + indices[2]) % 13377
            end

            y = y.map.with_index do |i, idx|
                indices = build_indices(idx)
                (i + indices[1]) % 13377
            end

            y = y.map.with_index do |i, idx|
                indices = build_indices(idx)
                (i + indices[3]) % 1337
            end

            y = y.map.with_index do |i, idx|
                indices = build_indices(idx)
                (i + indices[0]) % 13377
            end

            y = y.map.with_index do |i, idx|
                indices = build_indices(idx)
                (i + indices[1]) % 1377
            end
        end

        return y
    end

    def build_indices(idx)
        return (0...DIMS.size).map do |dim_index|
            index_div = DIMS.drop(dim_index + 1).reduce(1, :*)
            index_mod = DIMS[dim_index]

            if dim_index > 0
                (idx / index_div) % index_mod
            else
                idx / index_div
            end
        end
    end
end