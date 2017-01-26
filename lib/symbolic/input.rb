module Ikra
    module Symbolic
        # Specifies an input parameter for a parallel section. A parameter might be expanded to
        # multiple parameters in the generated CUDA code to avoid passing arrays etc.
        class Input
            # Returns the access pattern of this input, e.g., `:tid` (single element, identified
            # by thread ID) or `:entire` (access to entire array is necessary).
            attr_reader :pattern

            attr_reader :command

            def initialize(pattern:)
                # Currently supported: :tid, :entire
                @pattern = pattern
            end

            def ==(other)
                return self.class == other.class &&
                    self.pattern == other.pattern &&
                    self.command == other.command
            end

            def hash
                return (pattern.hash + command.hash) % 7656781
            end

            def eql?(other)
                return self == other
            end
        end

        # A single input value produced by one command.
        class SingleInput < Input
            def initialize(command:, pattern:)
                super(pattern: pattern)

                @command = command
            end
        end

        # An array containing values produced by one previous command.
        class StencilArrayInput < Input
            attr_reader :offsets
            attr_reader :out_of_bounds_value

            def initialize(command:, pattern:, offsets:, out_of_bounds_value:)
                super(pattern: pattern)

                @command = command
                @offsets = offsets
                @out_of_bounds_value = out_of_bounds_value
            end

            def ==(other)
                return super(other) && 
                    offsets == other.offsets &&
                    out_of_bounds_value == other.out_of_bounds_value
            end
        end

        class StencilSingleInput < Input
            attr_reader :offsets
            attr_reader :out_of_bounds_value

            def initialize(command:, pattern:, offsets:, out_of_bounds_value:)
                super(pattern: pattern)

                @command = command
                @offsets = offsets
                @out_of_bounds_value = out_of_bounds_value
            end

            def ==(other)
                return super(other) && 
                    offsets == other.offsets &&
                    out_of_bounds_value == other.out_of_bounds_value
            end
        end

        # Similar to [SingleInput], but two values are passed to the block.
        class ReduceInput < SingleInput

        end
    end
end

require_relative "input_visitor"
