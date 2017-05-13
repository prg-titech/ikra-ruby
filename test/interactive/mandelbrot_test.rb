# Rendering setup
require 'sdl'

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

def encodeHSBcolor(h, s, b)
  # h - the hue component, where (h - floor(h))*360 produce the hue angle
  # s - the saturation of the color     (0.0 <= s <= 1.0)
  # b - the brightness of the color     (0.0 <= b <= 1.0)
  c = (1 - (2*b - 1).abs)*s
  h_ = (h - h.floor)*360/60
  x = c * (1 - (h_ % 2 - 1).abs)
  if    h_ < 1 then r1 = c; g1 = x; b1 = 0
  elsif h_ < 2 then r1 = x; g1 = c; b1 = 0
  elsif h_ < 3 then r1 = 0; g1 = c; b1 = x
  elsif h_ < 4 then r1 = 0; g1 = x; b1 = c
  elsif h_ < 5 then r1 = x; g1 = 0; b1 = c
  else              r1 = c; g1 = 0; b1 = x
  end
  m = b - c/2
  r = r1 + m; g = g1 + m; b = b1 + m
  (r*255).to_i * 0x10000 + (g*255).to_i * 0x100 + (b*255).to_i
end

# Mandelbrot implementation
def mandelbrot_cpu(size)
    hx_res = size
    hy_res = size

    mandel_basic = Array.new(hx_res * hy_res) do |j|
        hx = j % hx_res
        hy = j / hx_res
        
        encodeHSBcolor(hx.to_f / hx_res, 1, 0.5)
    end

    return mandel_basic
end

def mandelbrot_gpu(size)
    hx_res = size
    hy_res = size

    mandel_basic = PArray.new(hx_res * hy_res) do |j|
        hx = j % hx_res
        hy = j / hx_res
        
        encodeHSBcolor(hx.to_f / hx_res, 1, 0.5)
    end

    return mandel_basic
end

# Program entry point
size = 1000

mode = ARGV[0]
if mode == "CPU"
    puts "CPU Rendering, wait..."
    show(size, size, mandelbrot_cpu(size).pack("I!*"))
elsif mode == "GPU"
    puts "GPU Rendering"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Rendering..."
    show(size, size, mandelbrot_gpu(size).pack("I!*"))
else
    puts "Invalid rendering mode: Pass CPU or GPU as argument"
end

puts "Press RETURN to continue"
STDIN.getc

