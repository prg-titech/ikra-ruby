# Rendering setup
require 'sdl'
require 'chunky_png'

require_relative '../../lib/ikra'

SCALE=0.2

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

def fusion_gpu_ikra_1
  Ikra::Configuration.kernel_iterations = 100

  hx_res = 4000
  hy_res = 4000

  base1 = Array.pnew(hx_res * hy_res) do |index|
    0x000000ff - index % 32
  end

  result = base1.pmap_with_index do |value, index|
    x = index%hx_res
    y = index/hx_res

    delta_x = hx_res/2 - x
    delta_y = hy_res/2 - y

    smaller_dim = hx_res < hy_res ? hx_res : hy_res

    if delta_x*delta_x + delta_y*delta_y < smaller_dim*smaller_dim/5
      value
    else
      0x00ff0000
    end
  end

  show(hx_res, hy_res, result.pack("I!*"))
end

def fusion_gpu_ikra_2
  Ikra::Configuration.kernel_iterations = 100

  hx_res = 4000
  hy_res = 4000

  base1 = Array.pnew(hx_res * hy_res) do |index|
    0x000000ff - index % 32
  end

  base2 = Array.pnew(hx_res * hy_res) do |index|
    x = index%hx_res
    encodeHSBcolor(x.to_f / hx_res, 1.0, 0.5)
  end.to_a

  result = base1.pmap_with_index do |value, index|
    x = index%hx_res
    y = index/hx_res

    delta_x = hx_res/2 - x
    delta_y = hy_res/2 - y

    smaller_dim = hx_res < hy_res ? hx_res : hy_res

    if delta_x*delta_x + delta_y*delta_y < smaller_dim*smaller_dim/5
      value
    else
      base2[index]
    end
  end

  show(hx_res, hy_res, result.pack("I!*"))
end


def custom_cuda_1
  Ikra::Configuration.override_cuda_file = "fusion_custom_1.cu"

  hx_res = 4000
  hy_res = 4000

  base1 = Array.pnew(hx_res * hy_res) do
    0x000000ff
  end

  result = base1.pmap_with_index do |value, index|
    x = index%hx_res
    y = index/hx_res

    delta_x = hx_res/2 - x
    delta_y = hy_res/2 - y

    smaller_dim = hx_res < hy_res ? hx_res : hy_res

    if delta_x*delta_x + delta_y*delta_y < smaller_dim*smaller_dim/5
      value
    else
      0
    end
  end

  show(hx_res, hy_res, result.pack("I!*"))
end

def custom_cuda_2
  Ikra::Configuration.override_cuda_file = "fusion_custom_2.cu"

  hx_res = 4000
  hy_res = 4000

  base1 = Array.pnew(hx_res * hy_res) do |index|
    0x000000ff - index % 32
  end

  base2 = Array.pnew(hx_res * hy_res) do |index|
    x = index%hx_res
    encodeHSBcolor(x.to_f / hx_res, 1.0, 0.5)
  end
  base2 = [1]

  Ikra::Configuration.kernel_iterations = 100

  result = base1.pmap_with_index do |value, index|
    x = index%hx_res
    y = index/hx_res

    delta_x = hx_res/2 - x
    delta_y = hy_res/2 - y

    smaller_dim = hx_res < hy_res ? hx_res : hy_res

    if delta_x*delta_x + delta_y*delta_y < smaller_dim*smaller_dim/5
      value
    else
      base2[index]
    end
  end

  show(hx_res, hy_res, result.pack("I!*"))
end

# Program entry point
mode = ARGV[0]
if mode == "GPU_1"
  puts "GPU Rendering, wait..."
  fusion_gpu_ikra_1
elsif mode == "GPU_2"
  puts "GPU Rendering, wait..."
  fusion_gpu_ikra_2
elsif mode == "CUSTOM_1"
  puts "Compile custom file ID: 1"
  custom_cuda_1
elsif mode == "CUSTOM_2"
  puts "Compile custom file ID: 2"
  custom_cuda_2
else
    puts "Invalid rendering mode"
end

puts "Press RETURN to continue"
STDIN.getc

