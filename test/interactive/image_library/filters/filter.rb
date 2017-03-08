module ImageLibrary
    module Filters
        class Filter
            attr_reader :block

            def initialize(&block)
                @block = block
            end
        end

        class StencilFilter < Filter
            attr_reader :neighborhood
            attr_reader :out_of_bounds_value

            def initialize(neighborhood:, out_of_bounds_value:, &block)
                super(&block)

                @neighborhood = neighborhood
                @out_of_bounds_value = out_of_bounds_value
            end

            def apply_to(command)
                return command.pstencil(neighborhood, out_of_bounds_value, &block)
            end
        end

        STENCIL_1 = [[0, 0]]

        STENCIL_2 = [
            [-1, -1], [0, -1], [1, -1], 
            [-1, 0], [0, 0], [1, 0], 
            [-1, 1], [0, 1], [1, 1]]
    end
end

# Add all filters
require_relative "helper.rb"
require_relative "stencils.rb"