
require "set"

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

class Array
    def to_type_array_string
        "[" + map do |set|
            set.to_type_string
        end.join(", ") + "]"
    end
end

class Set
    def to_type_string
        "{" + map do |type|
            type.to_s
        end.join(", ") + "}"
    end
end