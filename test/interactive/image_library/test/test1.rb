require_relative "../image_library"

a = ImageLibrary::Core.load_image("input.png")
a = a.apply_filter(ImageLibrary::Filters::Sharpen)

for i in 1..10
    a = a.apply_filter(ImageLibrary::Filters::Blur)
end

a.render
gets
