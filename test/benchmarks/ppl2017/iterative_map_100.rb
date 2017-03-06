require_relative "../benchmark_base"

class IterativeMap100 < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_command
            old_data = x

            for r in 0...500
                old_data = y
                y = y.pmap(with_index: true) do |i, indices|
                    ((i % 938) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
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
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        for r in 0...500
            base = base.map.with_index do |i, idx|
                indices = build_indices(idx)
                ((i % 938) + i / 97) % 97717 + (indices[indices[indices[i % 4] % 4] % 4] * (i % 7) % 99)
            end
        end

        return base
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