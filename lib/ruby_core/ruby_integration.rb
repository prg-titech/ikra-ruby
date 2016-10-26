require_relative "../types/union_type"

module Ikra
    module RubyIntegration
        class Implementation
            attr_reader :num_params
            attr_reader :implementation
            attr_reader :pass_self
            attr_reader :return_type

            def initialize(num_params:, return_type:, implementation:, pass_self: true)
                @num_params = num_params
                @implementation = implementation
                @pass_self = pass_self

                if return_type.is_a?(Symbol)
                    if return_type == :int
                        @return_type = Types::UnionType.create_int
                    elsif return_type == :float
                        @return_type = Types::UnionType.create_float
                    elsif return_type == :bool
                        @return_type = Types::UnionType.create_bool
                    else
                        raise "Unknown type shortcut: #{return_type}"
                    end
                else
                    @return_type = return_type
                end
            end
        end

        @@classes = {}
        @@classes.default_proc = proc do |hash, key|
            hash[key] = {}
        end

        def self.implement(class_, method_name, return_type, num_params, impl, pass_self: true)
            @@classes[class_][method_name] = Implementation.new(
                num_params: num_params,
                return_type: return_type,
                implementation: impl,
                pass_self: pass_self)
        end

        def self.has_implementation?(class_, method_name)
            return @@classes.include?(class_) && @@classes[class_].include?(method_name)
        end

        def self.should_pass_self?(class_, method_name)
            return @@classes[class_][method_name].pass_self
        end

        def self.get_implementation(class_, method_name, *args)
            source = @@classes[class_][method_name].implementation

            args.each_with_index do |arg, index|
                source.gsub!("##{index + 1}", arg)
            end

            return source
        end

        def self.get_return_type(class_, method_name)
            return @@classes[class_][method_name].return_type
        end
    end
end

require_relative "core"
require_relative "math"
