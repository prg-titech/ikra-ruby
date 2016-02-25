module Ikra
    module Translator
        class VariableWithType
            attr_reader :type
            attr_reader :var_name
            
            def initialize(type:, var_name:)
                @type = type
                @var_name = var_name
            end
        end

        class EnvironmentBuilder
            def initialize
                @vars = []
            end

            def add_variable(var_name:, type:, value:)
                @vars.push([var_name, type, value])
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