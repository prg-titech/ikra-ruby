require 'sdl'

module ImageLibrary
    module Core
        SCALE = 1.5

        class << self
            def setup(w,h)
                SDL.init(SDL::INIT_VIDEO)
                $screen = SDL::Screen.open(w*SCALE,h*SCALE,32,SDL::SWSURFACE)
            end

            def get_screen(w,h)
                $screen ||= setup(w,h)
            end

            def show_image(w, h, str)
                # to show a x*y pixels image constructed from str
                # w : width of the image, must be the multiple of 32
                # h : height of the image
                # str : a string of w*h characters, whose i'th character represents
                #       the color at (i%w, i/w).

                depth = 32
                pitch = w * 4
                surface = SDL::Surface.new_from(str, w, h, depth, pitch, 0,0,0,0)
                screen = get_screen(w,h)
                # screen.put(surface,0,0)

                SDL::Surface.transform_draw(surface,screen,
                    0,                  # angle
                    SCALE, SCALE,       # scale
                    0, 0, 0, 0,         # px, py, qx, qy
                    0)                  # flags
              screen.flip
            end
        end
    end
end

module Ikra
    module Symbolic
        module ArrayCommand
            def render
                ImageLibrary::Core.show_image(
                    self.dimensions[1], self.dimensions[0], to_a.pack("I!*"))
            end
        end
    end
end
