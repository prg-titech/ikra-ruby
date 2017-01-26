module Ikra
    module Symbolic
        module ArrayCommand
            include Types::RubyType

            # TODO: Either ensure that every type is a singleton or implement == properly
            
            def to_c_type
                return "#{result_type.to_c_type} *"
            end

            def to_ffi_type
                # TODO: This method is probably not required?
                return :pointer
            end

            def to_ruby_type
                return ArrayCommand
            end

            # Every [ArrayCommand] has itself as an Ikra type. This integrates well with the
            # current type inference approach and `ruby_core`.
            def ikra_type
                return self
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
