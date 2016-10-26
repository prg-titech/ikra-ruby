require "set"
require_relative "primitive_type"
require_relative "union_type"

class Fixnum
    class << self
        def to_ikra_type
            Ikra::Types::PrimitiveType::Int
        end
    end
end

class Float
    class << self
        def to_ikra_type
            Ikra::Types::PrimitiveType::Float
        end
    end
end

class TrueClass
    class << self
        def to_ikra_type
            Ikra::Types::PrimitiveType::Bool
        end
    end
end

class FalseClass
    class << self
        def to_ikra_type
            Ikra::Types::PrimitiveType::Bool
        end
    end
end
