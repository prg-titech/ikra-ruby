module ImageLibrary
    module Filters
        class CombineFilter < Filter
            attr_reader :args

            def initialize(*args, &block)
                super(&block)

                @args = args
            end

            def apply_to(command)
                return command.combine(*args, &block)
            end
        end

        def self.blend(other, ratio)
            return CombineFilter.new(other) do |p1, p2|
                s1 = pixel_scale(p1, 1.0 - ratio)
                s2 = pixel_scale(p2, ratio)

                pixel_add(s1, s2)
            end
        end

        def self.invert
            return CombineFilter.new do |p|
                r = pixel_get_r(p)
                b = pixel_get_b(p)
                g = pixel_get_g(p)

                build_pixel(255 - r, 255 - g, 255 - b)
            end
        end

        def self.overlay(mask_generator, overlay)
            mask = mask_generator.call(overlay.height, overlay.width)

            return CombineFilter.new(mask, overlay) do |b, m, o|
                # b, m, o: base, mask, overlay

                
                if m
                    o
                else
                    b
                end
            end
        end

    end
end
