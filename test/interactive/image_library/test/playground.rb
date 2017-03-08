require_relative "../image_library"

# Load Tokyo image
tokyo = ImageLibrary::Core.load_image("tokyo.png")

# Blur the image multiple times
for i in 1..10
    tokyo = tokyo.apply_filter(ImageLibrary::Filters.blur)
    #tokyo = tokyo.apply_filter(ImageLibrary::Filters.sharpen)
end

# Combine with sunset image
sunset = ImageLibrary::Core.load_image("sunset.png")
combined = tokyo.apply_filter(ImageLibrary::Filters.blend(sunset, 0.3))

#forest = ImageLibrary::Core.load_image("forest.png")
#forest = forest.apply_filter(ImageLibrary::Filters.invert) 
#combined = combined.apply_filter(ImageLibrary::Filters.blend(forest, 0.1))

# Show result
combined.render
gets
