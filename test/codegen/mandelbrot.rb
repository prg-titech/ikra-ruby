require "ikra"
require "chunky_png"

def run_mandel
    magnify = 1.0
    hx_res = 200
    hy_res = 200
    iter_max = 256

    mandel_basic = PArray.new(hx_res * hy_res) do |j|
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
                break
            end
        end
        
        iter % 256
    end

    inverted = 1
    mandel_filtered = mandel_basic.map do |color|
        if inverted == 1
            255 - color
        else
            color
        end
    end

    color_cache = {}
    color_cache.default_proc = Proc.new do |hash, key|
        hash[key] = ChunkyPNG::Color.rgb(key, 0, 0)
    end

    png = ChunkyPNG::Image.new(hx_res, hy_res, ChunkyPNG::Color::TRANSPARENT)
    for i in 0..mandel_filtered.size - 1
        png[i % hx_res, i / hx_res] = color_cache[mandel_filtered[i]]
    end

    png.save(Ikra::Configuration.codegen_expect_file_name_for("result.png"), :interlace => false)
end

