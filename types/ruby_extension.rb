require "set"
require_relative "primitive_type"
require_relative "dynamic_type"
require_relative "union_type"
require_relative "../translator"

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
        
        def _ikra_t_to_f(receiver_type)
            # TODO: check if this type can be cast
            UnionType.create_float
        end

        def _ikra_c_to_f(receiver)
            "(float) #{receiver}"
        end
    end
end

class Float
    class << self
        def to_ikra_type
            PrimitiveType::Float
        end
        
        def _ikra_c_to_f(receiver)
            Translator::TranslationResult.new(receiver.c_source, PrimitiveType::Float)
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