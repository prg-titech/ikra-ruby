module Ikra
    module AST
        class MethodDefinition
            def to_c_source
                method_params = (["#{type.to_c_type} _self_"] + parameter_types.map do |type|
                    type.singleton_type.to_c_type
                end.zip(parameter_names).map do |param|
                    "#{param[0]} #{param[1]}"
                end).join(", ")

                signature = "__device__ #{return_type.singleton_type.to_c_type} #{type.mangled_method_name(selector)}(#{method_params})"
                signature + "\n" + Translator.wrap_in_c_block(ast.translate_statement)
            end
        end
    end
end