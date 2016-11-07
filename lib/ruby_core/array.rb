module Ikra
    module RubyIntegration
        ALL_ARRAY_TYPES = proc do |type|
            type.is_a?(Types::ArrayType)
        end

        INNER_TYPE = proc do |rcvr|
            rcvr.inner_type
        end

        implement ALL_ARRAY_TYPES, :[], INNER_TYPE, 1, "#0[#I1]"
    end
end
