require_relative "../image_library"

# Load Tokyo image
tokyo = ImageLibrary::Core.load_image("tokyo.png")

# Blur the image multiple times
for i in 0...3
    tokyo = tokyo.apply_filter(ImageLibrary::Filters.blur)
    #tokyo = tokyo.apply_filter(ImageLibrary::Filters.sharpen)
end

# Combine with sunset image
sunset = ImageLibrary::Core.load_image("sunset.png")
combined = tokyo.apply_filter(ImageLibrary::Filters.blend(sunset, 0.3))

forest = ImageLibrary::Core.load_image("forest.png")
forest = forest.apply_filter(ImageLibrary::Filters.invert) 
#combined = combined.apply_filter(ImageLibrary::Filters.blend(forest, 0.1))

combined = combined.apply_filter(ImageLibrary::Filters.overlay(
    #ImageLibrary::Masks.rect(50, 0, 200, 20), forest))
    ImageLibrary::Masks.circle(tokyo.height / 4), forest))
    #ImageLibrary::Masks.mandelbrot, forest))

# Show result
combined.render
gets