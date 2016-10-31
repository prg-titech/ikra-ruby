
require "set"

module Ikra
    module Types

        # Defines the minimal interface for Ikra types. Instances of {UnionType} are expected in most cases.
        module RubyType
            @@next_class_id = 10

            def to_ruby_type
                raise NotImplementedError
            end
            
            def to_c_type
                raise NotImplementedError
            end
            
            def is_primitive?
                false
            end

            def is_union_type?
                false
            end

            def should_generate_self_arg?
                return true
            end

            def class_id
                if @class_id == nil
                    @class_id = @@next_class_id
                    @@next_class_id += 1
                end

                @class_id
            end
        end
    end
end

class Array
    def to_type_array_string
        "[" + map do |set|
            set.to_s
        end.join(", ") + "]"
    end
end
