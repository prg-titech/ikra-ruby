# No explicit `require`s. This file should be includes via types.rb

module Ikra
    module Types
        class ArrayType
            include RubyType

            class << self
                alias_method :new_original, :new

                # Ensure singleton per class
                def new(inner_type)
                    if @cache == nil
                        @cache = {}
                        @cache.default_proc = proc do |hash, key|
                            hash[key] = new_original(key)
                        end
                    end

                    return @cache[inner_type]
                end
            end

            attr_reader :inner_type
            
            def ==(other)
                return other.class == self.class && other.inner_type == self.inner_type
            end

            def initialize(inner_type)
                if not inner_type.is_union_type?
                    raise AssertionError.new("Union type expected")
                end

                @inner_type = inner_type
            end

            def to_s
                return "[#{self.class.to_s}, inner_type = #{inner_type}]"
            end

            def to_c_type
                return "#{@inner_type.to_c_type} *"
            end

            def to_ffi_type
                return :pointer
            end

            def to_ruby_type
                return ::Array
            end
        end

        class LocationAwareVariableSizeArrayType < ArrayType
            class << self
                def new(inner_type, location: :device)
                    if @cache == nil
                        @cache = {}
                        @cache.default_proc = Proc.new do |hash, key|
                            hash[key] = new_original(*key)
                        end
                    end

                    @cache[[inner_type, location]]
                end
            end

            # Determines if the array is allocated on the host or on the device
            attr_reader :location

            def ==(other)
                return super && self.location == other.location
            end

            def initialize(inner_type, location)
                @inner_type = inner_type
                @location = location
            end

            def to_c_type
                return "variable_size_array_t"
            end

            def to_command
                # No fusion possible here. The first parameter (target) is a reference to the
                # array command struct representing the [ArrayInHostSectionCommand].
                # TODO: The code depends on the template (variable name `cmd` and `input_0`).
                return Symbolic::ArrayInHostSectionCommand.new("cmd->input_0", @inner_type)
            end
        end
    end
end

class Array
    def ikra_type
        inner_type = Ikra::Types::UnionType.new

        self.each do |element|
            inner_type.add(element.ikra_type)
        end

        return Ikra::Types::ArrayType.new(inner_type)
    end
end