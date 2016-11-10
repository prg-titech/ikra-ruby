# Rendering setup
require 'sdl'
require 'chunky_png'

require_relative '../../lib/ikra'

SCALE=1

def setup(w,h)
  SDL.init(SDL::INIT_VIDEO)
  $screen = SDL::Screen.open(w*SCALE,h*SCALE,32,SDL::SWSURFACE)
end

def get_screen(w,h)
  $screen ||= setup(w,h)
end

def show(w,h,str)
  # to show a x*y pixels image constructed from str
  # w : width of the image, must be the multiple of 32
  # h : height of the image
  # str : a string of w*h characters, whose i'th character represents
  #       the color at (i%w, i/w).
  depth = 32
  pitch = w * 4
  surface = SDL::Surface.new_from(str, w, h, depth, pitch, 0,0,0,0)
  screen = get_screen(w,h)
#  screen.put(surface,0,0)
  SDL::Surface.transform_draw(surface,screen,
                              0,       # angle
                              SCALE,SCALE,     # scale
                              0,0,0,0, # px,py, qx,qy
                              0)       # flags
  screen.flip
end

def check_key
    while event = SDL::Event.poll
      case event
      when SDL::Event::Quit, SDL::Event::KeyDown
        exit
      end
    end
end

def check(wait=true)
  while true
    check_key
    if wait
      sleep 0.1
    else
      return
    end
  end
end

#  image = ChunkyPNG::Image.from_file('input2.png')
#  hx_res = image.width
#  hy_res = image.height
#
#  image_array = image.pixels.map do |value|
#    (value % 0xffffff00) >> 8
#  end

def encodeHSBcolor(h, s, b)
  # h - the hue component, where (h - floor(h))*360 produce the hue angle
  # s - the saturation of the color     (0.0 <= s <= 1.0)
  # b - the brightness of the color     (0.0 <= b <= 1.0)
  c = (1.0 - (2.0*b - 1.0).abs)*s
  h_ = (h - h.floor)*360.0/60
  x = c * (1.0 - (h_ % 2 - 1.0).abs)
  if    h_ < 1 then r1 = c; g1 = x; b1 = 0.0
  elsif h_ < 2 then r1 = x; g1 = c; b1 = 0.0
  elsif h_ < 3 then r1 = 0.0; g1 = c; b1 = x
  elsif h_ < 4 then r1 = 0.0; g1 = x; b1 = c
  elsif h_ < 5 then r1 = x; g1 = 0.0; b1 = c
  else              r1 = c; g1 = 0.0; b1 = x
  end
  m = b - c/2.0
  r = r1 + m; g = g1 + m; b = b1 + m
  (r*255).to_i * 0x10000 + (g*255).to_i * 0x100 + (b*255).to_i
end

class Ikra::Symbolic::ArrayNewCommand
  def apply(block)
    return pmap_with_index(&block)
  end
end

class Ikra::Symbolic::ArrayMapCommand
  def apply(block)
    return pmap_with_index(&block)
  end
end

class Ikra::Symbolic::ArrayIdentityCommand
  def apply(block)
    return pmap_with_index(&block)
  end
end

class Array
  def apply(block)
    return pmap_with_index(&block)
  end
end