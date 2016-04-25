
require "set"

module Ikra
    module Types

        # Defines the minimal interface for Ikra types. Instances of {UnionType} are expected in most cases.
        module RubyType
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
