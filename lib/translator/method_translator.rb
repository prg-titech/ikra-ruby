module Ikra
    module AST
        class MethDefNode
            def to_c_source
                # TODO: merge with BlockTranslator
                
                method_params = (["environment_t * #{Translator::Constants::ENV_IDENTIFIER}", "#{parent.get_type.to_c_type} #{Constants::SELF_IDENTIFIER}"] + parameters_names_and_types.map do |name, type|
                    "#{name} #{type.singleton_type.to_c_type}"
                end).join(", ")

                # TODO: load environment variables

                # Declare local variables
                local_variables_def = ""
                local_variables_names_and_types.each do |name, type|
                    local_variables_def += "#{types.to_c_type} #{name};\n"
                end

                signature = "__device__ #{get_type.singleton_type.to_c_type} #{parent.get_type.mangled_method_name(name)}(#{method_params})"
                signature + "\n" + Translator.wrap_in_c_block(local_variables_def + body.translate_statement)
            end
        end
    end
end