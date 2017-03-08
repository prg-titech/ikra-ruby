require 'chunky_png'

module ImageLibrary
    module Core
        class PNGImage
            attr_reader :width
            attr_reader :height
            attr_reader :pixels

            def initialize(width:, height:, pixels:)
                @width = width
                @height = height
                @pixels = pixels
            end
        end

        class << self
            def read_png(file_name)
                image = ChunkyPNG::Image.from_file(file_name)
                hx_res = image.width
                hy_res = image.height

                pixels = image.pixels.map do |value|
                  (value % 0xffffff00) >> 8
                end

                return PNGImage.new(
                    width: hx_res,
                    height: hy_res,
                    pixels: pixels)
            end
        end
    end
end
