module Ikra
    module AST
        class MethodDefinition
            def to_c_source
                method_params = (["#{type.to_c_type} #{Constants::SELF_IDENTIFIER}"] + parameter_variables.map do |name, type|
                    "#{name} #{type.singleton_type.to_c_type}"
                end).join(", ")

                signature = "__device__ #{return_type.singleton_type.to_c_type} #{type.mangled_method_name(selector)}(#{method_params})"
                signature + "\n" + Translator.wrap_in_c_block(ast.translate_statement)
            end
        end
    end
end