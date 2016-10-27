# Rendering setup
require 'sdl'
require 'chunky_png'

require_relative '../../lib/cpu/cpu_implementation'

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

def rect_neighborhood(hx_res, size)
  offsets = Array.new(size * size)
  index = 0

  for y in -size..size
    for x in -size..size
      offsets[index] = y * hx_res + x
      index += 1
    end
  end

  return offsets
end

# http://setosa.io/ev/image-kernels/
# Stencil implementation
def stencil_cpu1(hx_res, hy_res, base)
    filtered = base.stencil(rect_neighborhood(hx_res, 1), 0) do |p1, p2, p3, p4, p5, p6, p7, p8, p9|
      b1 = (p1 & 0xff0000) >> 16
      b2 = (p2 & 0xff0000) >> 16
      b3 = (p3 & 0xff0000) >> 16
      b4 = (p4 & 0xff0000) >> 16
      b5 = (p5 & 0xff0000) >> 16
      b6 = (p6 & 0xff0000) >> 16
      b7 = (p7 & 0xff0000) >> 16
      b8 = (p8 & 0xff0000) >> 16
      b9 = (p9 & 0xff0000) >> 16

      r1 = (p1 & 0x00ff00) >> 8
      r2 = (p2 & 0x00ff00) >> 8
      r3 = (p3 & 0x00ff00) >> 8
      r4 = (p4 & 0x00ff00) >> 8
      r5 = (p5 & 0x00ff00) >> 8
      r6 = (p6 & 0x00ff00) >> 8
      r7 = (p7 & 0x00ff00) >> 8
      r8 = (p8 & 0x00ff00) >> 8
      r9 = (p9 & 0x00ff00) >> 8

      g1 = p1 & 0x0000ff
      g2 = p2 & 0x0000ff
      g3 = p3 & 0x0000ff
      g4 = p4 & 0x0000ff
      g5 = p5 & 0x0000ff
      g6 = p6 & 0x0000ff
      g7 = p7 & 0x0000ff
      g8 = p8 & 0x0000ff
      g9 = p9 & 0x0000ff

      r = (-r1+0*r2+r3-2*r4+0*r5+2*r6-r7+0*r8+r9) 
      g = (-g1+0*g2+g3-2*g4+0*g5+2*g6-g7+0*g8+g9) 
      b = (-b1+0*b2+b3-2*b4+0*b5+2*b6-b7+0*b8+b9) 

      (b << 16) + (r << 8) + g
    end

    return filtered
end

# Stencil implementation
def stencil_cpu2(hx_res, hy_res, base)
    filtered = base.stencil(rect_neighborhood(hx_res, 1), 0) do |p1, p2, p3, p4, p5, p6, p7, p8, p9|
      b1 = (p1 & 0xff0000) >> 16
      b2 = (p2 & 0xff0000) >> 16
      b3 = (p3 & 0xff0000) >> 16
      b4 = (p4 & 0xff0000) >> 16
      b5 = (p5 & 0xff0000) >> 16
      b6 = (p6 & 0xff0000) >> 16
      b7 = (p7 & 0xff0000) >> 16
      b8 = (p8 & 0xff0000) >> 16
      b9 = (p9 & 0xff0000) >> 16

      r1 = (p1 & 0x00ff00) >> 8
      r2 = (p2 & 0x00ff00) >> 8
      r3 = (p3 & 0x00ff00) >> 8
      r4 = (p4 & 0x00ff00) >> 8
      r5 = (p5 & 0x00ff00) >> 8
      r6 = (p6 & 0x00ff00) >> 8
      r7 = (p7 & 0x00ff00) >> 8
      r8 = (p8 & 0x00ff00) >> 8
      r9 = (p9 & 0x00ff00) >> 8

      g1 = p1 & 0x0000ff
      g2 = p2 & 0x0000ff
      g3 = p3 & 0x0000ff
      g4 = p4 & 0x0000ff
      g5 = p5 & 0x0000ff
      g6 = p6 & 0x0000ff
      g7 = p7 & 0x0000ff
      g8 = p8 & 0x0000ff
      g9 = p9 & 0x0000ff

      r = (r1+r2+r3+r4+r5+r6+r7+r8+r9) / 9
      b = (b1+b2+b3+b4+b5+b6+b7+b8+b9) / 9
      g = (g1+g2+g3+g4+g5+g6+g7+g8+g9) / 9

      (b << 16) + (r << 8) + g
    end

    return filtered
end

# Stencil implementation
def stencil_cpu3(hx_res, hy_res, base)
    filtered = base.stencil(rect_neighborhood(hx_res, 1), 0) do |p1, p2, p3, p4, p5, p6, p7, p8, p9|
      b1 = (p1 & 0xff0000) >> 16
      b2 = (p2 & 0xff0000) >> 16
      b3 = (p3 & 0xff0000) >> 16
      b4 = (p4 & 0xff0000) >> 16
      b5 = (p5 & 0xff0000) >> 16
      b6 = (p6 & 0xff0000) >> 16
      b7 = (p7 & 0xff0000) >> 16
      b8 = (p8 & 0xff0000) >> 16
      b9 = (p9 & 0xff0000) >> 16

      r1 = (p1 & 0x00ff00) >> 8
      r2 = (p2 & 0x00ff00) >> 8
      r3 = (p3 & 0x00ff00) >> 8
      r4 = (p4 & 0x00ff00) >> 8
      r5 = (p5 & 0x00ff00) >> 8
      r6 = (p6 & 0x00ff00) >> 8
      r7 = (p7 & 0x00ff00) >> 8
      r8 = (p8 & 0x00ff00) >> 8
      r9 = (p9 & 0x00ff00) >> 8

      g1 = p1 & 0x0000ff
      g2 = p2 & 0x0000ff
      g3 = p3 & 0x0000ff
      g4 = p4 & 0x0000ff
      g5 = p5 & 0x0000ff
      g6 = p6 & 0x0000ff
      g7 = p7 & 0x0000ff
      g8 = p8 & 0x0000ff
      g9 = p9 & 0x0000ff

      r = (-r2-r4+5*r5-r6-r8) 
      b = (-b2-b4+5*b5-b6-b8) 
      g = (-g2-g4+5*g5-g6-g8) 

      (b << 16) + (r << 8) + g
    end

    return filtered
end


def stencil_gpu(hx_res, hy_res, base)
    filtered = base.pstencil(rect_neighborhood(hx_res, 1), 0) do |p1, p2, p3, p4, p5, p6, p7, p8, p9|
      b1 = (p1 & 0xff0000) >> 16
      b2 = (p2 & 0xff0000) >> 16
      b3 = (p3 & 0xff0000) >> 16
      b4 = (p4 & 0xff0000) >> 16
      b5 = (p5 & 0xff0000) >> 16
      b6 = (p6 & 0xff0000) >> 16
      b7 = (p7 & 0xff0000) >> 16
      b8 = (p8 & 0xff0000) >> 16
      b9 = (p9 & 0xff0000) >> 16

      r1 = (p1 & 0x00ff00) >> 8
      r2 = (p2 & 0x00ff00) >> 8
      r3 = (p3 & 0x00ff00) >> 8
      r4 = (p4 & 0x00ff00) >> 8
      r5 = (p5 & 0x00ff00) >> 8
      r6 = (p6 & 0x00ff00) >> 8
      r7 = (p7 & 0x00ff00) >> 8
      r8 = (p8 & 0x00ff00) >> 8
      r9 = (p9 & 0x00ff00) >> 8

      g1 = p1 & 0x0000ff
      g2 = p2 & 0x0000ff
      g3 = p3 & 0x0000ff
      g4 = p4 & 0x0000ff
      g5 = p5 & 0x0000ff
      g6 = p6 & 0x0000ff
      g7 = p7 & 0x0000ff
      g8 = p8 & 0x0000ff
      g9 = p9 & 0x0000ff

      r = (r1+r2+r3+r4+r5+r6+r7+r8+r9) / 9
      b = (b1+b2+b3+b4+b5+b6+b7+b8+b9) / 9
      g = (g1+g2+g3+g4+g5+g6+g7+g8+g9) / 9

      (b << 16) + (r << 8) + g
    end

    return filtered
end

# Program entry point
size = 1000

image = ChunkyPNG::Image.from_file('input3.png')
hx_res = image.width
hy_res = image.height

base = image.pixels.map do |value|
  (value % 0xffffff00) >> 8
end


mode = ARGV[0]
if mode == "CPU"
    puts "CPU Rendering, wait..."
    img = base

    for i in 1..1
      img = stencil_cpu1(hx_res, hy_res, img)
      show(hx_res, hy_res, img.pack("I!*"))

    end

    
elsif mode == "GPU"
    puts "GPU Rendering"
    puts "Initializing Ikra..."
    require_relative '../../lib/ikra'

    puts "Rendering..."
    show(hx_res, hy_res, stencil_gpu(hx_res, hy_res, base).pack("I!*"))
else
    puts "Invalid rendering mode: Pass CPU or GPU as argument"
end

puts "Press RETURN to continue"
STDIN.getc

