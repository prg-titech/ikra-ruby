require_relative "ruby_type"

class PrimitiveType
    include RubyType
    
    attr_reader :c_size
    
    def initialize(c_type, ruby_type, c_size)
        @c_type = c_type
        @ruby_type = ruby_type
        @c_size = c_size
    end
    
    Int = self.new("int", Fixnum, 4)
    Float = self.new("float", Float, 4)
    Bool = self.new("bool", TrueClass, 1)
    Void = self.new("void", nil, 0)
    
    def to_ruby_type
        @ruby_type
    end
    
    def to_c_type
        @c_type
    end
    
    def is_primitive?
        true
    end
    
    def to_s
        "<primitive: #{@c_type}>"
    end
end

class Fixnum
    def self.to_ikra_type
        PrimitiveType::Int
    end
end

class Float
    def self.to_ikra_type
        PrimitiveType::Float
    end
end

class TrueClass
    def self.to_ikra_type
        PrimitiveType::Bool
    end
end

class FalseClass
    def self.to_ikra_type
        PrimitiveType::Bool
    end
end
