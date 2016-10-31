require "test/unit"
require "ikra"
require_relative "codegen/mandelbrot"
require_relative "codegen/square"
require_relative "codegen/union_type_float_int_benchmark"

class CodegenTest < Test::Unit::TestCase
    def setup
        Ikra::Configuration.reset_state

        test_name = self.class.to_s + "\#" + method_name
        file_name = Ikra::Configuration.log_file_name_for(test_name)
        File.delete(file_name) if File.exist?(file_name)
        Ikra::Log.reopen(file_name)
    end

    def test_mandelbrot
        Ikra::Configuration.codegen_expect_file_name = "mandelbrot"
        run_mandel
    end

    def test_square
        Ikra::Configuration.codegen_expect_file_name = "square"
        run_square
    end

    def test_union_singleton_benchmark
        Ikra::Configuration.codegen_expect_file_name = "union_type_float_int_benchmark"
        run_gradient_benchmark
    end
end