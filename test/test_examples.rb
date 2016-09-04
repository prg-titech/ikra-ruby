require "test/unit"
require "ikra"

class ExamplesTest < Test::Unit::TestCase
	def test_mandelbrot
		Ikra::Configuration.codegen_expect_file_name = "mandelbrot"
		require_relative "examples/mandelbrot"
	end
end