require_relative "ruby_type"
require "set"

class UnionType
    include RubyType

    attr_reader :types

    def initialize(*types)
        @types = Set.new(types)
    end

    def to_c_type
        if @types.size == 1
            @types.first.to_c_type
        else
            "union_t"
        end
    end

    def to_ruby_type
        if @types.size == 1
            @types.first.to_ruby_type
        else
            raise "Not implemented yet"
        end
    end

    def is_primitive?
        if @types.size == 1
            @types.first.is_primitive?
        else
            false
        end
    end

    def is_union_type?
        true
    end

    def is_singleton?
        @types.size == 1
    end

    def singleton_type
        if @types.size != 1
            raise "Union type is not singleton"
        end

        @types.first
    end

    def expand(union_type)
        if not union_type.is_union_type?
            raise "Cannot expand with non-union type"
        end

        is_expanded = (union_type.types - @types).size > 0
        @types.merge(union_type.types)
        is_expanded
    end

    def expand_return_type(union_type)
        expand(union_type)
        self
    end

    def include?(singleton_type)
        if singleton_type.is_union_type?
            raise "Union type can never be included in union type"
        end

        @types.include?(singleton_type)
    end

    def include_all?(union_type)
        if not union_type.is_union_type?
            raise "Union type expected"
        end

        (union_type.types - @types).size == 0
    end

    # subseteq
    def <=(union_type)
        union_type.include_all?(self)
    end

    def to_s
        "{#{@types.to_a.join(", ")}}"
    end

    class << self
        def create_int
            new(PrimitiveType::Int)
        end

        def create_float
            new(PrimitiveType::Float)
        end

        def create_bool
            new(PrimitiveType::Bool)
        end

        def create_void
            new(PrimitiveType::Void)
        end

        def create_unknown
            new(UnknownType::Instance)
        end

        def array_subseteq(subseq_type, superset_type)
            subseteq.size == superseteq.type and
                subseteq.zip(superseteq).map do |tuple|
                    if not tuple[0].is_union_type? or tuple[1].is_union_type?
                        raise "Union type expected"
                    end

                    tuple[0] <= tuple[1]
                end.reduce(:&)
        end

        def parameter_hash_to_s(hash)
            hash.map do |name, type|
                name.to_s + ": " + type.to_s
            end.join(", ")
        end
    end
end