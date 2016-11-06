require_relative "../types/types"

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

        @@impls = {}
        @@impls.default_proc = proc do |hash, key|
            hash[key] = {}
        end

        def self.implement(rcvr_type, method_name, return_type, num_params, impl, pass_self: true)
            @@impls[rcvr_type][method_name] = Implementation.new(
                num_params: num_params,
                return_type: return_type,
                implementation: impl,
                pass_self: pass_self)
        end

        def self.has_implementation?(rcvr_type, method_name)
            return find_impl(rcvr_type, method_name) != nil
        end

        def self.should_pass_self?(rcvr_type, method_name)
            return find_impl(rcvr_type, method_name).pass_self
        end

        # Returns the implementation (CUDA source code snippet) for a method with name 
        # [method_name] defined on [rcvr_type]. Arguments in [arg] is the code snippet
        # for the receiver followed by type-snippet pairs for additional arguments.
        def self.get_implementation(method_name, args_types, args_code)
            rcvr_type = args_types.first
            impl = find_impl(rcvr_type, method_name)
            source = impl.implementation

            if source.is_a?(Proc)
                source = source.call(args_types, args_code)
            end

            if impl.pass_self
                substitution_args = args_code
            else
                # Do not pass `self`: Omit first argument
                substitution_args = args_code[1..-1]
            end

            substitution_args.each_with_index do |arg, index|
                source = source.gsub("\##{index + 1}", arg)
            end
            
            return source
        end

        def self.get_return_type(rcvr_type, method_name, *arg_types)
            return_type = find_impl(rcvr_type, method_name).return_type
            num_params = find_impl(rcvr_type, method_name).num_params

            if return_type.is_a?(Proc)
                # Return type depends on argument types
                if num_params != arg_types.size
                    raise "#{num_params} arguments expected"
                else
                    return return_type.call(rcvr_type, *arg_types)
                end
            else
                return return_type
            end
        end

        private

        def self.find_impl(rcvr_type, method_name)
            if @@impls.include?(rcvr_type) && @@impls[rcvr_type].include?(method_name)
                return @@impls[rcvr_type][method_name]
            else
                # Evaluate blocks
                for type_or_block in @@impls.keys
                    if type_or_block.is_a?(Proc)
                        if type_or_block.call(rcvr_type)
                            if @@impls[type_or_block].include?(method_name)
                                return @@impls[type_or_block][method_name]
                            end
                        end
                    end
                end

                # No implementation found
                return nil
            end
        end
    end
end

require_relative "core"
require_relative "math"
require_relative "array"