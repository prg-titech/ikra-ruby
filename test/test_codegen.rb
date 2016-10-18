require "test/unit"
require "ikra"
require_relative "codegen/mandelbrot"
require_relative "codegen/square"

class CodegenTest < Test::Unit::TestCase
    def setup
        Ikra::Configuration.reset_state
    end

    def test_mandelbrot
        Ikra::Configuration.codegen_expect_file_name = "mandelbrot"
        run_mandel
    end

    def test_square
        Ikra::Configuration.codegen_expect_file_name = "square"
        run_square
    end
end