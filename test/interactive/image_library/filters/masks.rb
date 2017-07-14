require_relative "../../../../lib/ikra"

module ImageLibrary
    module Masks

        # Rectangle mask
        def self.rect(x1, y1, x2, y2)
            return proc do |height, width|
                PArray.new(dimensions: [height, width]) do |indices|
                    y = indices[0]
                    x = indices[1]

                    if x >= x1 && y >= y1 && x <=x2 && y < y2
                        # Inside
                        true
                    else
                        # Outside
                        false
                    end
                end
            end
        end

        # Circle mask
        def self.circle(radius)
            return proc do |height, width|
                PArray.new(dimensions: [height, width]) do |indices|
                    y = indices[0]
                    x = indices[1]

                    x_diff = width / 2 - x
                    y_diff = height / 2 - y

                    if x_diff * x_diff + y_diff * y_diff < radius * radius
                        # Inside
                        true
                    else
                        # Outside
                        false
                    end
                end
            end
        end

        # Mandelbrot mask
        def self.mandelbrot
            return proc do |height, width|
                magnify = 1.0
                hx_res = width
                hy_res = height
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
                    
                    (iter % 256) == 0
                end
            end
        end

    end
end