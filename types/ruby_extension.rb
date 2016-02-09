require_relative "primitive_type"
require_relative "dynamic_type"

class Object
    class << self
        def to_ikra_type
            DynamicType
        end
    end
end

class Fixnum
    class << self
        def to_ikra_type
            PrimitiveType::Int
        end
    end
end

class Float
    class << self
        def to_ikra_type
            PrimitiveType::Float
        end
    end
end

class TrueClass
    class << self
        def to_ikra_type
            PrimitiveType::Bool
        end
    end
end

class FalseClass
    class << self
        def to_ikra_type
            PrimitiveType::Bool
        end
    end
end