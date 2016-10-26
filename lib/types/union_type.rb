require_relative "ruby_type"
require_relative "primitive_type"
require "set"

module Ikra
    module Types

        # Represents a possibly polymorphic type consisting of 0..* instances of {RubyType}. Only primitive types or class types are allowed for these inner types, but not union types.
        class UnionType
            include RubyType

            # @return [Set<RubyType>] Inner types
            attr_reader :types

            def initialize(*types)
                # Check arguments
                types.each do |type|
                    if type.is_union_type?
                        raise "Union type not allowed as inner type of union type"
                    end
                end

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

            # Returns the single inner type or raises an error if this union type contains more than one inner type.
            # @return [RubyType] Inner type
            def singleton_type
                if @types.size != 1
                    raise "Union type is not singleton"
                end

                @types.first
            end

            # Merges all inner types of the argument into this union type.
            # @param [UnionType] union_type The other union type
            # @return [Bool] true if the argument added new inner types to this union type
            def expand(union_type)
                if not union_type.is_union_type?
                    raise "Cannot expand with non-union type"
                end

                is_expanded = (union_type.types - @types).size > 0
                @types.merge(union_type.types)
                is_expanded
            end

            # Merges all inner types of the argument into this union type.
            # @param [UnionType] union_type The other union type
            # @return [UnionType] self
            def expand_return_type(union_type)
                expand(union_type)
                self
            end

            def expand_with_singleton_type(singleton_type)
                if singleton_type.is_union_type?
                    raise "Singleton type expected"
                end

                is_expanded = !@types.include?(singleton_type)
                @types.add(singleton_type)
                is_expanded
            end

            # Determines if this union type contains a specific singleton type.
            # @param [RubyType] singleton_type The other type
            # @return [Bool] true if the type is included
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

            # Alias for {#include_all?}
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

                def parameter_hash_to_s(hash)
                    hash.map do |name, type|
                        name.to_s + ": " + type.to_s
                    end.join(", ")
                end
            end
        end
    end
end