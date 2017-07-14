require_relative "../benchmark_base"

class Himeno < Test::Unit::TestCase
    include BenchmarkBase

    DIMS_A = [129, 65, 65, 4]
    DIMS_B = [129, 65, 65, 3]
    DIMS_C = [129, 65, 65, 3]
    DIMS_BND = [129, 65, 65]
    DIMS_WRK = [129, 65, 65]

    def build_indices(idx, dims)
        return (0...dims.size).map do |dim_index|
            index_div = dims.drop(dim_index + 1).reduce(1, :*)
            index_mod = dims[dim_index]

            if dim_index > 0
                (idx / index_div) % index_mod
            else
                idx / index_div
            end
        end
    end

    def prepare_matrices
        puts "Preparing matrices"

        i_max = 129
        j_max = 65
        k_max = 65  

        #i_max = 4
        #j_max = 5
        #k_max = 6 

        # Initialize matrices
        @param_a = Array.new([i_max, j_max, k_max, 4].reduce(:*)) do |i|
            indices = build_indices(i, DIMS_A)

            if indices[3] == 3
                1.0 / 6.0
            else
                0.0
            end
        end

        @param_b = Array.new([i_max, j_max, k_max, 3].reduce(:*)) do |i|
            0.0
        end

        @param_c = Array.new([i_max, j_max, k_max, 3].reduce(:*)) do |i|
            1.0
        end

        @param_bnd = Array.new([i_max, j_max, k_max].reduce(:*)) do |i|
            1.0
        end

        @param_wrk = Array.new([i_max, j_max, k_max].reduce(:*)) do |i|
            0.0
        end
    end

    def initialize(*args)
        super
        prepare_matrices
    end

    def execute
        i_max = 129
        j_max = 65
        k_max = 65  

        #i_max = 4
        #j_max = 5
        #k_max = 6  


        param_a = @param_a
        param_b = @param_b
        param_c = @param_c
        param_bnd = @param_bnd
        param_wrk = @param_wrk

        base_p = PArray.new(dimensions: [i_max, j_max, k_max]) do |indices|
            k = indices[2]
            #(k * k).to_f / (k_max - 1) / (k_max - 1)
            (k * k).to_f / (65 - 1) / (65 - 1)
        end

        section_result = Ikra::Symbolic.host_section(base_p) do |base|
            next_p = base.__call__.to_pa
            old_data = base
            old_old_data = base

            for r in 0...1000
                old_old_data = old_data
                old_data = next_p

                next_p = next_p.stencil([
                    [0, 0, 0],
                    [1, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 1], 
                    [1, 1, 0], 
                    [1, -1, 0], 
                    [-1, 1, 0], 
                    [-1, -1, 0],
                    [0, 1, 1],
                    [0, -1, 1],
                    [0, 1, -1],
                    [0, -1, -1],
                    [1, 0, 1],
                    [-1, 0, 1],
                    [1, 0, -1],
                    [-1, 0, -1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]], 0, with_index: true) do |values, indices|


                    #idx_a_0 = 65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 0
                    #idx_a_1 = 65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 1
                    #idx_a_2 = 65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 2
                    #idx_a_3 = 65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 3

                    #idx_b_0 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0
                    #idx_b_1 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1
                    #idx_b_2 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2

                    #idx_c_0 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0
                    #idx_c_1 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1
                    #idx_c_2 = 65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2

                    #idx_wrk = 65 * 65 * indices[0] + 65 * indices[1] + indices[2]

                    #s0 = param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 0] * values[1][0][0] + param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 1] * values[0][1][0] + param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 2] * values[0][0][1] + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * (values[1][1][0] - values[1][-1][0] - values[-1][1][0] + values[-1][-1][0] ) + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * (values[0][1][1] - values[0][-1][+1] - values[0][1][-1] + values[0][-1][-1] ) + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * (values[1][0][1] - values[-1][0][1] - values[1][0][-1] + values[-1][0][-1] ) + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * values[-1][0][0] + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * values[0][-1][0] + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * values[0][0][-1] + param_wrk[65 * 65 * indices[0] + 65 * indices[1] + indices[2]]

                    #ss = ((param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 0] * values[1][0][0] + param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 1] * values[0][1][0] + param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 2] * values[0][0][1] + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * (values[1][1][0] - values[1][-1][0] - values[-1][1][0] + values[-1][-1][0] ) + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * (values[0][1][1] - values[0][-1][+1] - values[0][1][-1] + values[0][-1][-1] ) + param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * (values[1][0][1] - values[-1][0][1] - values[1][0][-1] + values[-1][0][-1] ) + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * values[-1][0][0] + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * values[0][-1][0] + param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * values[0][0][-1] + param_wrk[65 * 65 * indices[0] + 65 * indices[1] + indices[2]]) * param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 3] - values[0][0][0]) * param_bnd[65 * 65 * indices[0] + 65 * indices[1] + indices[2]]

                    # TODO: Generate checksum (?)
                    # gosa = gosa + ss*ss;


                     values[0][0][0] + 0.8 * (((
                        param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 0] * values[1][0][0] + 
                        
                        param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 1] * values[0][1][0] + 
                        
                        param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 2] * values[0][0][1] + 
                        
                        param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * 
                            (values[1][1][0] - values[1][-1][0] - values[-1][1][0] + values[-1][-1][0] ) + 

                        param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * 
                            (values[0][1][1] - values[0][-1][+1] - values[0][1][-1] + values[0][-1][-1] ) + 

                        param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * 
                            (values[1][0][1] - values[-1][0][1] - values[1][0][-1] + values[-1][0][-1] ) + 

                        param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * values[-1][0][0] + 

                        param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * values[0][-1][0] + 

                        param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * values[0][0][-1] + 

                        param_wrk[65 * 65 * indices[0] + 65 * indices[1] + indices[2]]) * 

                        param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 3] - 

                        values[0][0][0]) * param_bnd[65 * 65 * indices[0] + 65 * indices[1] + indices[2]])


                end

                if r > 1
                    old_old_data.free_memory
                end
            end

            next_p
        end

        return section_result

        #assert_equal([0, 501, 1002, 5010, 5511, 6012], section_result.to_a)
    end
end

"""

 values[0][0][0] + 0.8 * (((
    param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 0] * values[1][0][0] + 
    
    param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 1] * values[0][1][0] + 
    
    param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 2] * values[0][0][1] + 
    
    param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * 
        (values[1][1][0] - values[1][-1][0] - values[-1][1][0] + values[-1][-1][0] ) + 

    param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * 
        (values[0][1][1] - values[0][-1][+1] - values[0][1][-1] + values[0][-1][-1] ) + 

    param_b[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * 
        (values[1][0][1] - values[-1][0][1] - values[1][0][-1] + values[-1][0][-1] ) + 

    param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 0] * values[-1][0][0] + 

    param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 1] * values[0][-1][0] + 

    param_c[65 * 65 * 3 * indices[0] + 65 * 3 * indices[1] + 3 * indices[2] + 2] * values[0][0][-1] + 

    param_wrk[65 * 65 * indices[0] + 65 * indices[1] + indices[2]]) * 

    param_a[65 * 65 * 4 * indices[0] + 65 * 4 * indices[1] + 4 * indices[2] + 3] - 

    values[0][0][0]) * param_bnd[65 * 65 * indices[0] + 65 * indices[1] + indices[2]])

    """