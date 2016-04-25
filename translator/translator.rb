require "ffi"

module Ikra
    module Translator
        class EnvironmentBuilder
            def initialize
                @vars = []
            end

            def add_variable(var_name:, type:, value:)
                if not type.is_union_type?
                    raise "Expected union type"
                end

                @vars.push([var_name, type, value])
            end

            def to_ffi_struct_type
                env_struct_layout = []
                env_index = 0
                @vars.each do |var|
                    env_struct_layout += [:"field_#{env_index}", var[1].singleton_type.to_ffi_type]
                    env_index += 1
                end

                env_struct_type = Class.new(FFI::Struct)
                env_struct_type.layout(*env_struct_layout)

                env_struct_type
            end

            def get_var(index)
                @vars[index][2]
            end

            def size
                @vars.size
            end

            def c_size
                size = 0
                @vars.each do |var|
                    size += var.singleton_type.c_size
                end

                size
            end

            def struct_definition(struct_name)
                defs = @vars.map do |var|
                    "#{var[1].singleton_type.to_c_type} #{var[0]};"
                end.join("\n")

                """struct #{struct_name}
#{wrap_in_c_block(defs)};

"""
            end

            def wrap_in_c_block(str)
                "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}"
            end
        end

        EnvParameterName = "_env_"
        EnvStructName = "Environment"

        class << self
            def translation_id
                @translation_id || 0
            end

            def increase_translation_id
                @translation_id ||= 0
                @translation_id += 1
            end

            def mangle_name_translation_id(str)
                "_k_#{translation_id}_#{str}"
            end

            def mangle_block_name_translation_id(str)
                "_block_k_#{translation_id}_#{str}"
            end

            def mangle_var_name_translation_id(str)
                "_var_k_#{translation_id}_#{str}"
            end

            def wrap_in_c_block(str)
                "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
            end
        end
    end
end