# No explicit `require`s. This file should be includes via types.rb

require "forwardable"
require "set"

module Ikra
    module Types

        # Represents a possibly polymorphic type consisting of 0..* instances of {RubyType}. Only 
        # primitive types or class types are allowed for these inner types, but not union types.
        class UnionType
            include RubyType
            include Enumerable
            extend Forwardable

            def_delegator :@types, :each

            # @return [Set<RubyType>] Inner types
            attr_accessor :types

            def ==(other)
                return other.class == self.class && other.types == self.types
            end

            def dup
                result = super
                result.types = @types.dup
                return result
            end
            
            def initialize(*types)
                # Check arguments
                types.each do |type|
                    if type.is_union_type?
                        raise AssertionError.new(
                            "Union type not allowed as inner type of union type")
                    end
                end

                @types = Set.new(types)
            end

            def empty?
                return @types.empty?
            end

            # Removes all type information.
            def clear!
                @types.clear
            end

            # Removes all singleton types contained in `union_type` from this type.
            def remove!(union_type)
                if union_type.is_singleton?
                    raise AssertionError.new("Union type expected")
                else
                    @types.delete_if do |type|
                        union_type.include?(type)
                    end
                end
            end

            def size
                return @types.size
            end

            def to_c_type
                if is_singleton?
                    return @types.first.to_c_type
                else
                    return "union_t"
                end
            end

            def c_size
                if is_singleton?
                    return @types.first.c_size
                else
                    return "sizeof(union_t)"
                end
            end

            def to_ruby_type
                if is_singleton?
                    return @types.first.to_ruby_type
                else
                    raise NotImplementedError.new
                end
            end

            def to_ffi_type
                if is_singleton?
                    return @types.first.to_ffi_type
                else
                    raise NotImplementedError.new
                end
            end

            def is_primitive?
                if is_singleton?
                    return @types.first.is_primitive?
                else
                    return false
                end
            end

            def is_union_type?
                return true
            end

            def to_union_type
                return self
            end

            def is_singleton?
                return @types.size == 1
            end

            # Returns the single inner type or raises an error if this union type contains more than one inner type.
            # @return [RubyType] Inner type
            def singleton_type
                if !is_singleton?
                    raise AssertionError.new(
                        "Union type is not singleton (found #{@types.size} types)")
                end

                return @types.first
            end

            # Merges all inner types of the argument into this union type.
            # @param [UnionType] union_type The other union type
            # @return [Bool] true if the argument added new inner types to this union type
            def expand(union_type)
                if not union_type.is_union_type?
                    raise AssertionError.new(
                        "Cannot expand with non-union type: #{union_type}")
                end

                is_expanded = false

                for type in union_type
                    is_expanded = is_expanded | add(type)
                end

                return is_expanded
            end

            # Merges all inner types of the argument into this union type.
            # @param [UnionType] union_type The other union type
            # @return [UnionType] self
            def expand_return_type(union_type)
                expand(union_type)
                return self
            end

            # Adds a singleton type to this union type.
            # @return True if the type was extended
            def add(singleton_type)
                if singleton_type.is_union_type?
                    raise AssertionError.new("Singleton type expected")
                end

                if singleton_type == PrimitiveType::Int && include?(PrimitiveType::Float)
                    # Special rule: Coerce int to float
                    return false
                elsif singleton_type == PrimitiveType::Float && include?(PrimitiveType::Int)
                    # Special rule: Coerce int to float
                    @types.delete(PrimitiveType::Int)
                    @types.add(singleton_type)
                    return true
                else
                    is_expanded = !@types.include?(singleton_type)
                    @types.add(singleton_type)
                    return is_expanded
                end
            end

            # Determines if this union type contains a specific singleton type.
            # @param [RubyType] singleton_type The other type
            # @return [Bool] true if the type is included
            def include?(singleton_type)
                if singleton_type.is_union_type?
                    raise AssertionError.new("Union type can never be included in union type")
                end

                @types.include?(singleton_type)
            end

            def include_all?(union_type)
                if not union_type.is_union_type?
                    raise AssertionError.new("Union type expected")
                end

                return (union_type.types - @types).size == 0
            end

            # Alias for {#include_all?}
            def <=(union_type)
                return union_type.include_all?(self)
            end

            def to_s
                return "U{#{@types.to_a.join(", ")}}"
            end

            class << self
                def create_int
                    return new(PrimitiveType::Int)
                end

                def create_float
                    return new(PrimitiveType::Float)
                end

                def create_bool
                    return new(PrimitiveType::Bool)
                end

                def create_void
                    return new(PrimitiveType::Void)
                end

                def create_nil
                    return new(PrimitiveType::Nil)
                end

                def create_unknown
                    return new(UnknownType::Instance)
                end

                def parameter_hash_to_s(hash)
                    return hash.map do |name, type|
                        name.to_s + ": " + type.to_s
                    end.join(", ")
                end
            end
        end
    end
end