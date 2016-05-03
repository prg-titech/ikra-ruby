require "ffi"

module Ikra
    module Translator
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