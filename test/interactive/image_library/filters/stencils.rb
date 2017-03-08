module ImageLibrary
    module Filters
        class StencilFilter < Filter
            attr_reader :neighborhood
            attr_reader :out_of_bounds_value

            def initialize(neighborhood:, out_of_bounds_value:, &block)
                super(&block)

                @neighborhood = neighborhood
                @out_of_bounds_value = out_of_bounds_value
            end

            def apply_to(command)
                return command.pstencil(neighborhood, out_of_bounds_value, &block)
            end
        end

        STENCIL_1 = [[0, 0]]

        STENCIL_2 = [
            [-1, -1], [0, -1], [1, -1], 
            [-1, 0], [0, 0], [1, 0], 
            [-1, 1], [0, 1], [1, 1]]

        def self.identity_stencil
            return StencilFilter.new(
                neighborhood: STENCIL_2,
                out_of_bounds_value: 0) do |values|

                # TODO: Support single expr without begin node
                values[0][0] + 0
            end
        end

        def self.blur
            return StencilFilter.new(
                neighborhood: STENCIL_2,
                out_of_bounds_value: 0) do |values|

                factor = 1.0/9.0
                r = pixel_get_r(values[-1][-1]) + pixel_get_r(values[-1][0]) + pixel_get_r(values[-1][1]) + pixel_get_r(values[0][-1]) + pixel_get_r(values[0][0]) + pixel_get_r(values[0][1]) + pixel_get_r(values[1][-1]) + pixel_get_r(values[1][0]) + pixel_get_r(values[1][1])
                g = pixel_get_g(values[-1][-1]) + pixel_get_g(values[-1][0]) + pixel_get_g(values[-1][1]) + pixel_get_g(values[0][-1]) + pixel_get_g(values[0][0]) + pixel_get_g(values[0][1]) + pixel_get_g(values[1][-1]) + pixel_get_g(values[1][0]) + pixel_get_g(values[1][1])
                b = pixel_get_b(values[-1][-1]) + pixel_get_b(values[-1][0]) + pixel_get_b(values[-1][1]) + pixel_get_b(values[0][-1]) + pixel_get_b(values[0][0]) + pixel_get_b(values[0][1]) + pixel_get_b(values[1][-1]) + pixel_get_b(values[1][0]) + pixel_get_b(values[1][1])

                build_pixel((factor * r).to_i, (factor * g).to_i, (factor * b).to_i)
            end
        end

        def self.sharpen
            return StencilFilter.new(
                neighborhood: STENCIL_2,
                out_of_bounds_value: 0) do |values|

                factor = 1.0
                r = 0 - pixel_get_r(values[-1][0]) - pixel_get_r(values[0][-1]) + 5 * pixel_get_r(values[0][0]) - pixel_get_r(values[0][1]) - pixel_get_r(values[1][0])
                g = 0 - pixel_get_g(values[-1][0]) - pixel_get_g(values[0][-1]) + 5 * pixel_get_g(values[0][0]) - pixel_get_g(values[0][1]) - pixel_get_g(values[1][0])
                b = 0 - pixel_get_b(values[-1][0]) - pixel_get_b(values[0][-1]) + 5 * pixel_get_b(values[0][0]) - pixel_get_b(values[0][1]) - pixel_get_b(values[1][0])

                build_pixel((factor * r).to_i, (factor * g).to_i, (factor * b).to_i)
            end
        end
    end
end
