module ImageLibrary
    module Filters
        IdentityStencil = StencilFilter.new(
            neighborhood: STENCIL_2,
            out_of_bounds_value: 0) do |values|

            values[0][0] + 0
        end

        Blur = StencilFilter.new(
            neighborhood: STENCIL_2,
            out_of_bounds_value: 0) do |values|

            factor = 1.0/9.0
            r = pixel_get_r(values[-1][-1]) + pixel_get_r(values[-1][0]) + pixel_get_r(values[-1][1]) + pixel_get_r(values[0][-1]) + pixel_get_r(values[0][0]) + pixel_get_r(values[0][1]) + pixel_get_r(values[1][-1]) + pixel_get_r(values[1][0]) + pixel_get_r(values[1][1])
            g = pixel_get_g(values[-1][-1]) + pixel_get_g(values[-1][0]) + pixel_get_g(values[-1][1]) + pixel_get_g(values[0][-1]) + pixel_get_g(values[0][0]) + pixel_get_g(values[0][1]) + pixel_get_g(values[1][-1]) + pixel_get_g(values[1][0]) + pixel_get_g(values[1][1])
            b = pixel_get_b(values[-1][-1]) + pixel_get_b(values[-1][0]) + pixel_get_b(values[-1][1]) + pixel_get_b(values[0][-1]) + pixel_get_b(values[0][0]) + pixel_get_b(values[0][1]) + pixel_get_b(values[1][-1]) + pixel_get_b(values[1][0]) + pixel_get_b(values[1][1])

            build_pixel((factor * r).to_i, (factor * g).to_i, (factor * b).to_i)
        end

        Sharpen = StencilFilter.new(
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
