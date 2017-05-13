require_relative "../benchmark_base"

class IterativeStencil100NoHostSection < Test::Unit::TestCase
    include BenchmarkBase

    DIMS = [20, 500, 500, 2]

    def execute
        Ikra::Translator::CommandTranslator::KernelLauncher.debug_free_previous_input_immediately = true

        base = PArray.new(dimensions: DIMS) do |indices|
            (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)
        end

        y = base


        for r in 0...200
            y = y.stencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
            ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
            end
        end

        return y
    end

    def expected
        # Just reseting debug flag
        Ikra::Translator::CommandTranslator::KernelLauncher.debug_free_previous_input_immediately = false
        return nil
    end
end