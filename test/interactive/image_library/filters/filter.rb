module ImageLibrary
    module Filters
        class Filter
            attr_reader :block

            def initialize(&block)
                @block = block
            end
        end
    end
end

# Add all filters
require_relative "helper.rb"
require_relative "stencils.rb"
require_relative "maps.rb"