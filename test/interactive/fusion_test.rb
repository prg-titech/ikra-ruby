require_relative "fusion_setup"


# AcmeImgLib: A modular image manipulation library.
module AcmeImgLib

  class << self
    # Loads a PNG file from the file system.
    def load_image(filename)
      # Load the picture
      image = ChunkyPNG::Image.from_file(filename)
      hx_res = image.width
      hy_res = image.height

      return image.pixels.map do |value|
        (value % 0xffffff00) >> 8
      end, hx_res, hy_res
    end

    # Converts all pixels to grayscale
    def grayscale_filter
      return proc do |value, index|
        r = (value & 0x00ff0000) >> 16
        g = (value & 0x0000ff00) >> 8
        b = value & 0x000000ff

        average = (r + g + b) / 3

        (average << 16) + (average << 8) + average
      end
    end

    # Adds another image on top of the manipulated image (outside of the area of a circle).
    def overlay_outside_circle(overlay, hx_res, hy_res, padding: 0)
      overlay_array = overlay.to_a

      return proc do |value, index|
        x = index%hx_res
        y = index/hx_res

        delta_x = hx_res/2 - x
        delta_y = hy_res/2 - y

        if hx_res < hy_res
          smaller_dim = hx_res
        else
          smaller_dim = hy_res
        end

        if delta_x*delta_x + delta_y*delta_y < smaller_dim*smaller_dim/(4 + padding)
          value
        else
          overlay_array[index]
        end
      end
    end

    # Blend with another image, i.e., for every pixel, take the average values of red, green, and blue.
    def blend(image)
      image_array = image.to_a

      return proc do |value, index|
        # TODO: Your implementation here
        # Blend `image_array[index]` with `value`

        # Remove the following code, this is just here for demonstration purposes
        img_width = 1024  # Image width hard-coded
        y = index/img_width
        block_height = 50

        if (y/50) % 2 == 0
          value
        else
          image_array[index]
        end
      end
    end

    # Generates a horizontal gradient.
    def horizontal_gradient(hx_res, hy_res)
      return Array.pnew(hx_res * hy_res) do |index|
        x = index%hx_res
        encodeHSBcolor(x.to_f / hx_res, 1.0, 0.5)
      end
    end

    # Generates a vertical gradient.
    def vertical_gradient(hx_res, hy_res)
      return Array.pnew(hx_res * hy_res) do |index|
        y = index/hx_res
        encodeHSBcolor(y.to_f / hy_res, 1.0, 0.5)
      end
    end
  end

end

def fusion_gpu_ikra_1
  # Let the kernel run 100 times to isolate the effect of global memory access
  # You can reduce this number or use a different (smaller) picture to speedup rendering
  Ikra::Configuration.kernel_iterations = 100

  # Load the picture
  image, hx_res, hy_res = AcmeImgLib.load_image('input2.png')

  # Convert to grayscale
  image = image.apply(AcmeImgLib.grayscale_filter)

  # Generate gradients
  h_gradient = AcmeImgLib.horizontal_gradient(hx_res, hy_res)
  v_gradient = AcmeImgLib.vertical_gradient(hx_res, hy_res)

  # Blend picture with gradient
  filter_blend = AcmeImgLib.blend(v_gradient)
  image = image.apply(filter_blend)

  # Overlay outer gradient
  filter_outer_gradient = AcmeImgLib.overlay_outside_circle(h_gradient, hx_res, hy_res)
  image = image.apply(filter_outer_gradient)

  # Show the picture
  show(hx_res, hy_res, image.pack("I!*"))
end


def custom_cuda_1
  Ikra::Configuration.override_cuda_file = "fusion_custom_1.cu"

  # Load the picture
  image, hx_res, hy_res = AcmeImgLib.load_image('input2.png')

  # Convert to grayscale
  result = image.pmap_with_index do |value, index|
    1 # DUMMY -- Actual source code read from file
  end

  show(hx_res, hy_res, result.pack("I!*"))
end

# Program entry point
mode = ARGV[0]
if mode == "GPU_1"
  puts "GPU Rendering, wait..."
  fusion_gpu_ikra_1
elsif mode == "CUSTOM_1"
  puts "Compile custom file ID: 1"
  custom_cuda_1
else
    puts "Invalid rendering mode"
end

puts "Press RETURN to continue"
STDIN.getc
