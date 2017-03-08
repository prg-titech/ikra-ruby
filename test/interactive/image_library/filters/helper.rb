# Helper functions
# TODO: Fix nesting
def pixel_get_r(value)
    return (value & 0x00ff0000) >> 16
end

def pixel_get_g(value)
    return (value & 0x0000ff00) >> 8
end

def pixel_get_b(value)
    return value & 0x000000ff
end

def build_pixel(r, g, b)
    rr = r
    gg = g
    bb = b

    if rr < 0
        rr = 0
    end
    if rr > 255
        rr = 255
    end
    
    if gg < 0
        gg = 0
    end
    if gg > 255
        gg = 255
    end

    if bb < 0
        bb = 0
    end
    if bb > 255
        bb = 255
    end

    return (rr << 16) + (gg << 8) + bb
end

def pixel_scale(pixel, factor)
    r = pixel_get_r(pixel)
    g = pixel_get_g(pixel)
    b = pixel_get_b(pixel)

    return build_pixel((r * factor).to_i, (g * factor).to_i, (b * factor).to_i)
end

def pixel_add(p1, p2)
    r1 = pixel_get_r(p1)
    g1 = pixel_get_g(p1)
    b1 = pixel_get_b(p1)

    r2 = pixel_get_r(p2)
    g2 = pixel_get_g(p2)
    b2 = pixel_get_b(p2)

    r = r1 + r2
    g = g1 + g2
    b = b1 + b2

    return build_pixel(r, g, b)
end
