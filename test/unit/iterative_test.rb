require "ikra"
require_relative "unit_test_template"

class IterativeTest < UnitTestCase
    ITERATIONS = 100

    def test_simple_update
        array = Array.pnew(100) do |j|
            j
        end

        iterative_update = Ikra::Symbolic::IterativeCommandWrapper.new(array).pmap do |i|
            i + 1
        end

        iterative_computation = Ikra::Symbolic::IterativeComputation.new(
            updates: {array: iterative_update},
            until_condition: ITERATIONS)

        array_cpu = Array.new(100) do |j|
            j + ITERATIONS
        end

        assert_equal(array_cpu, iterative_computation[:array].to_a)
    end
end