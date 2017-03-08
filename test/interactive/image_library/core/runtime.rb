require_relative "../../../../lib/ikra"
require_relative "file_io"
require_relative "output"

module ImageLibrary
    module Core
        class << self
            def load_image(file_name)
                image = read_png(file_name)

                pixels = image.pixels
                width = image.width
                height = image.height

                return Array.pnew(dimensions: [height, width]) do |indices|
                    one_d_index = indices[0] * width + indices[1]
                    pixels[one_d_index]
                end
            end
        end
    end
end

module Ikra
    module Symbolic
        module ArrayCommand
            def apply_filter(filter)
                filter.apply_to(self)
            end

            def render
                ImageLibrary::Core.show_image(
                    self.dimensions[1], self.dimensions[0], to_a.pack("I!*"))
            end
        end
    end
end
