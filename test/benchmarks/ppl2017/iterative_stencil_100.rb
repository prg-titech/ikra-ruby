require_relative "../benchmark_base"

class IterativeStencil100 < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        base = PArray.new(dimensions: DIMS) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x.__call__.to_pa
            old_data = x
            old_old_data = x

            for r in 0...200
                old_old_data = old_data
                old_data = y

                y = y.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
                ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
                end

                if r > 1
                    old_old_data.free_memory
                end
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

        for i in 0...10
            new_base = Array.new(DIMS.reduce(:*)) do |idx|
                indices = build_indices(idx)

                # Build neighborhood indices
                n_0 = indices.dup
                n_0[0] = n_0[0] - 1

                n_1 = indices.dup

                n_2 = indices.dup
                n_2[0] = n_2[0] + 1

                n_3 = indices.dup
                n_3[0] = n_3[0] - 1
                n_3[1] = n_3[1] - 1

                n_all = [n_0, n_1, n_2, n_3]

                # Check if out of bounds
                out_of_bounds = false
                for n_i in n_all
                    (0...(DIMS.size)).each do |dim_index|
                        if n_i[dim_index] < 0 || n_i[dim_index] >= DIMS[dim_index]
                            out_of_bounds = true
                        end
                    end
                end

                if out_of_bounds
                    37
                else
                    # Get values
                    v_0 = base[build_1d_index(n_0)]
                    v_1 = base[build_1d_index(n_1)]
                    v_2 = base[build_1d_index(n_2)]
                    v_3 = base[build_1d_index(n_3)]

                    ((v_0 % 938) + v_1 / 97) % 97717 + (indices[indices[indices[v_2 % 4] % 4] % 4] * (v_3 % 7) % 99)
                end
            end

            base = new_base
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

    def build_1d_index(indices)
        multiplier = 1
        result = 0

        (0...(DIMS.size)).reverse_each do |dim_index|
            result = result + indices[dim_index] * multiplier
            multiplier = multiplier * DIMS[dim_index]
        end

        return result
    end
end