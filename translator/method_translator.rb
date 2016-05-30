module Ikra
    module AST
        class MethodDefinition
            def to_c_source
                # TODO: merge with BlockTranslator
                
                method_params = (["environment_t * #{Translator::Constants::ENV_IDENTIFIER}", "#{type.to_c_type} #{Constants::SELF_IDENTIFIER}"] + parameter_variables.map do |name, type|
                    "#{name} #{type.singleton_type.to_c_type}"
                end).join(", ")

                # TODO: load environment variables

                # Declare local variables
                local_variables_def = ""
                local_variables.each do |name, types|
                    local_variables_def += "#{types.singleton_type.to_c_type} #{name};\n"
                end

                signature = "__device__ #{return_type.singleton_type.to_c_type} #{type.mangled_method_name(selector)}(#{method_params})"
                signature + "\n" + Translator.wrap_in_c_block(local_variables_def + ast.translate_statement)
            end
        end
    end
end