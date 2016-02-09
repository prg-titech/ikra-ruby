require_relative "ruby_type"

class PrimitiveType
    include RubyType
    
    def initialize(c_type, ruby_type)
        @c_type = c_type
        @ruby_type = ruby_type
    end
    
    Int = self.new("int", Fixnum)
    Float = self.new("float", Float)
    Bool = self.new("bool", TrueClass)
    Void = self.new("void", nil)
    
    def to_ruby_type
        @ruby_type
    end
    
    def to_c_type
        @c_type
    end
    
    def is_primitive?
        true
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
