def encodeHSBcolorSingleton(h, s, b)
  # h - the hue component, where (h - floor(h))*360 produce the hue angle
  # s - the saturation of the color     (0.0 <= s <= 1.0)
  # b - the brightness of the color     (0.0 <= b <= 1.0)
  c = (1 - (2*b - 1).abs)*s
  h_ = (h - h.floor)*360/60
  x = c * (1 - (h_ % 2 - 1).abs)
  if    h_ < 1 then r1 = c; g1 = x; b1 = 0.0
  elsif h_ < 2 then r1 = x; g1 = c; b1 = 0.0
  elsif h_ < 3 then r1 = 0.0; g1 = c; b1 = x
  elsif h_ < 4 then r1 = 0.0; g1 = x; b1 = c
  elsif h_ < 5 then r1 = x; g1 = 0.0; b1 = c
  else              r1 = c; g1 = 0.0; b1 = x
  end
  m = b - c/2
  r = r1 + m; g = g1 + m; b = b1 + m
  (r*255).to_i * 0x10000 + (g*255).to_i * 0x100 + (b*255).to_i
end

def encodeHSBcolorUnion(h, s, b)
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

# Note: This example does not have branch divergence
def gradient_union(size)
    hx_res = size
    hy_res = size

    command = Array.pnew(hx_res * hy_res) do |j|
        hx = j % hx_res
        hy = j / hx_res
        
        encodeHSBcolorUnion(hx.to_f / hx_res, 1, 0.5)
    end

    command.execute
end

def gradient_singleton(size)
    hx_res = size
    hy_res = size

    command = Array.pnew(hx_res * hy_res) do |j|
        hx = j % hx_res
        hy = j / hx_res
        
        encodeHSBcolorSingleton(hx.to_f / hx_res, 1, 0.5)
    end

    command.execute
end

def run_gradient_benchmark
    size = 10500
    gradient_union(size)
    gradient_singleton(size)
end
