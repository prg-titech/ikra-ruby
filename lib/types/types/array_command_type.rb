require_relative "../../custom_weak_cache"

module Ikra
    module Symbolic
        module ArrayCommand
            include Types::RubyType

            def self.included(base)
                base.extend(ClassMethods)
            end

            module ClassMethods
                # Ensure that ArrayCommands are singletons. Otherwise, we have a problem, because
                # two equal commands can have different class IDs.
                def new(*args, **kwargs, &block)
                    if @cache == nil
                        @cache = CustomWeakCache.new
                    end

                    new_command = super

                    if @cache.include?(new_command)
                        return @cache.get_value(new_command)
                    else
                        @cache.add_value(new_command)
                        return new_command
                    end
                end
            end

            def to_c_type
                return "#{Translator::ArrayCommandStructBuilder.struct_name(self)} *"
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
