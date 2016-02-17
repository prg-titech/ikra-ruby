require_relative "symbolic"
require "chunky_png"

magnify = 1.0
hx_res = 500
hy_res = 500
iter_max = 100

result = Array.pnew(hx_res * hy_res) do |j|
    hx = j % hx_res
    hy = j / hx_res
    
    cx = (hx.to_f / hx_res.to_f - 0.5) / magnify*3.0 - 0.7
    cy = (hy.to_f / hy_res.to_f - 0.5) / magnify*3.0
    
    x = 0.0
    y = 0.0
    
    for iter in 0..iter_max
        xx = x*x - y*y + cx
        y = 2.0*x*y + cy
        x = xx
        
        if x*x + y*y > 100
            iter = 999
            break
        end
    end
    
    if iter == 999
        0
    else
        1
    end
end

png = ChunkyPNG::Image.new(hx_res, hy_res, ChunkyPNG::Color::TRANSPARENT)
for i in 0..result.size - 1
    png[i % hx_res, i / hx_res] = result[i] == 0 ? ChunkyPNG::Color('blue') : ChunkyPNG::Color('white')
end

png.save('result.png', :interlace => true)

