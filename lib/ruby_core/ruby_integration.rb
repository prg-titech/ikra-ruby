require_relative "../types/types"

module Ikra
    module RubyIntegration
        INT = Types::UnionType.create_int
        FLOAT = Types::UnionType.create_float
        BOOL = Types::UnionType.create_bool

        INT_S = INT.singleton_type
        FLOAT_S = FLOAT.singleton_type
        BOOL_S = BOOL.singleton_type

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
                sub_code = args_code
                sub_types = args_types
            else
                # Do not pass `self`: Omit first argument
                sub_code = args_code[1..-1]
                sub_types = args_types[1..-1]
            end

            sub_indices = (0...source.length).find_all do |index| 
                source[index] == "#" 
            end
            substitutions = {}
            sub_indices.each do |index|
                if source[index + 1] == "F"
                    # Insert float
                    arg_index = source[index + 2].to_i

                    if arg_index >= sub_code.size
                        raise "Argument missing: Expected at least #{arg_index + 1}, found #{sub_code.size}"
                    end

                    substitutions["\#F#{arg_index}"] = code_argument(FLOAT_S, sub_types[arg_index], sub_code[arg_index])
                elsif source[index + 1] == "I"
                    # Insert integer
                    arg_index = source[index + 2].to_i

                    if arg_index >= sub_code.size
                        raise "Argument missing: Expected at least #{arg_index + 1}, found #{sub_code.size}"
                    end

                    substitutions["\#I#{arg_index}"] = code_argument(INT_S, sub_types[arg_index], sub_code[arg_index])
                elsif source[index + 1] == "B"
                    # Insert integer
                    arg_index = source[index + 2].to_i

                    if arg_index >= sub_code.size
                        raise "Argument missing: Expected at least #{arg_index + 1}, found #{sub_code.size}"
                    end

                    substitutions["\#B#{arg_index}"] = code_argument(BOOL_S, sub_types[arg_index], sub_code[arg_index])
                elsif source[index + 1] == "N"
                    # Numeric, coerce integer to float
                    arg_index = source[index + 2].to_i

                    if arg_index >= sub_code.size
                        raise "Argument missing: Expected at least #{arg_index + 1}, found #{sub_code.size}"
                    end

                    if sub_types[arg_index].include?(FLOAT_S)
                        expected_type = FLOAT_S
                    else
                        expected_type = INT_S
                    end

                    substitutions["\#N#{arg_index}"] = code_argument(expected_type, sub_types[arg_index], sub_code[arg_index])
                else
                    arg_index = source[index + 1].to_i

                    if arg_index >= sub_code.size
                        raise "Argument missing: Expected at least #{arg_index + 1}, found #{sub_code.size}"
                    end

                    substitutions["\##{arg_index}"] = sub_code[arg_index]
                end
            end

            substitutions.each do |key, value|
                # Do not use `gsub!` here!
                source = source.gsub(key, value)
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

        def self.code_argument(expected_type, arg_type, code)
            if arg_type.is_singleton?
                if expected_type != arg_type.singleton_type
                    # Try to cast
                    return "((#{expected_type.to_c_type}) #{code})"
                else
                    return code
                end
            else
                # Extract from union type
                result = StringIO.new

                result << "({ union_t arg = #{code};\n"
                result << "    #{expected_type.to_c_type} result;\n"
                result << "    switch (arg.class_id) {\n"

                for type in arg_type
                    c_type = expected_type.to_c_type
                    result << "        case #{type.class_id}:\n"
                    # TODO: This works only for primitive types
                    result << "            result = (#{c_type}) arg.value.#{type.to_c_type}_;\n"
                    result << "            break;\n"
                end

                result << "        default:\n"
                result << "            // TODO: throw exception\n"
                result << "    }\n"
                result << "    result;\n"
                result << "})"

                return result.string
            end
        end

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