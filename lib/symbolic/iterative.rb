module Ikra
    module Symbolic

        # This is a wrapper for [ArrayCommand] updates. It is used for specifying that an
        # operations is meant be applied iteratively, starting with a certain value (array).
        class IterativeCommandWrapper
            include ParallelOperations

            attr_reader :command

            # Find the base of this wrapper recursively.
            #
            # A wrapper can wrap any kind of command. A command can have more than one input. In
            # such a case, only the first input may be considered as a base.
            #
            # E.g.: base.pcombine(b2, b3) do ... end
            # In this example, we update `base`, but not `b2` and `b3`, even though both of them
            # are also being read and used as input.
            def find_base
                if base?
                    return self
                else
                    # First input must be an [IterativeCommandWrapper] again by definition
                    return command.input.first.find_base
                end
            end

            def base?
                return @is_base
            end

            def initialize(command, is_base: true)
                @command = command
                @is_base = is_base
            end

            # Required only for [ParallelOperations]
            def to_command
                return self
            end

            def method_missing(symbol, *args, **kwargs, &block)
                return_value = command.send(symbol, *args, **kwargs, &block)

                if return_value.is_a?(ArrayCommand)
                    # Wrap command
                    return IterativeCommandWrapper.new(return_value, is_base: false)
                else
                    return return_value
                end
            end

            def execute
                raise "Iterative commands cannot be executed directly"
            end
        end

        class IterativeComputation
            attr_reader :updates
            attr_reader :until_condition

            def initialize(updates:, until_condition:)
                @updates = updates
                @until = until_condition
            end

            # Accesses the result of this iterative computation. Returns an [ArrayCommand].
            def [](key)
                return IterativeComputationResultCommand.new(
                    command: @updates[key],
                    iterative_computation: self)
            end

            class IterativeComputationResultCommand
                include ArrayCommand

                attr_reader :command
                attr_reader :iterative_computation

                def initialize(command:, iterative_computation:)
                    @command = command
                    @iterative_computation = iterative_computation
                end

                def size
                    return @command.size
                end
            end
        end
    end
end