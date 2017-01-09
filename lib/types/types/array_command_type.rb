module Ikra
    module Types
        class ArrayCommandType
            include RubyType

            attr_reader :size
            attr_reader :inner_type

            def initialize(size:, inner_type:)
                @size = size
                @inner_type = inner_type
            end

            def to_c_type
                return "#{@inner_type.to_c_type} *"
            end

            def to_ffi_type
                return :pointer
            end

            def to_ruby_type
                return Symbolic::ArrayCommand
            end
        end
    end

    module Symbolic
        module ArrayCommand
            def self.to_ikra_type_obj(object)
                # Perform type inference (if not done yet)
                return Types::ArrayCommandType.new(
                    size: object.size,
                    inner_type: object.result_type)
            end

            def result_type
                # Result cache should be cached, just like the result itself
                if @result_type == nil
                    @result_type = TypeInference::CommandInference.process_command(self)
                end

                return @result_type
            end
        end
    end
end
