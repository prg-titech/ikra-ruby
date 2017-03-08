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

                return pixels.to_command(dimensions: [image.height, image.width])
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
        end
    end
end
