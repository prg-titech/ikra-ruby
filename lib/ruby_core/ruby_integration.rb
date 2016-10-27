require_relative "../types/union_type"

module Ikra
    module RubyIntegration
        INT = Types::UnionType.create_int
        FLOAT = Types::UnionType.create_float
        BOOL = Types::UnionType.create_bool

        class Implementation
            attr_reader :num_params
            attr_reader :implementation
            attr_reader :pass_self
            attr_reader :return_type

            def initialize(num_params:, return_type:, implementation:, pass_self: true)
                @num_params = num_params
                @implementation = implementation
                @pass_self = pass_self
                @return_type = return_type
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
                source = source.gsub("\##{index + 1}", arg)
            end
            
            return source
        end

        def self.get_return_type(class_, method_name, *arg_types)
            return_type = @@classes[class_][method_name].return_type
            num_params = @@classes[class_][method_name].num_params

            if return_type.is_a?(Proc)
                # Return type depends on argument types
                if num_params != arg_types.size
                    raise "#{num_params} arguments expected"
                else
                    return return_type.call(*arg_types)
                end
            else
                return return_type
            end
        end
    end
end

require_relative "core"
require_relative "math"
