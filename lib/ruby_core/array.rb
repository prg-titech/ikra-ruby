module Ikra
    module RubyIntegration
        ALL_ARRAY_TYPES = proc do |type|
            type.is_a?(Types::ArrayType) && !type.is_a?(Types::LocationAwareVariableSizeArrayType)
        end

        LOCATION_AWARE_VARIABLE_ARRAY_SIZE_TYPE = proc do |type|
            # TODO: Maybe there should be an automated transfer to host side here if necessary?
            type.is_a?(Types::LocationAwareVariableSizeArrayType)
        end

        LOCATION_AWARE_VARIABLE_ARRAY_SIZE_ACCESS = proc do |receiver, method_name, args, translator, result_type|

            recv = receiver.accept(translator.expression_translator)
            inner_type = receiver.get_type.singleton_type.inner_type.to_c_type
            index = args[0].accept(translator.expression_translator)

            "((#{inner_type} *) #{recv}.content)[#{index}]"
        end

        INNER_TYPE = proc do |rcvr|
            rcvr.inner_type
        end

        implement ALL_ARRAY_TYPES, :[], INNER_TYPE, 1, "#0[#I1]"

        implement(
            LOCATION_AWARE_VARIABLE_ARRAY_SIZE_TYPE, 
            :[], 
            INNER_TYPE, 
            1, 
            LOCATION_AWARE_VARIABLE_ARRAY_SIZE_ACCESS)
    end
end
