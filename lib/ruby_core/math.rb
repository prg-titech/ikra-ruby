module Ikra
    module RubyIntegration
        implement Math.singleton_class, :cos, FLOAT, 1, "cosf((float) #1)", pass_self: false
        implement Math.singleton_class, :sin, FLOAT, 1, "sinf((float) #1)", pass_self: false
        implement Math.singleton_class, :tan, FLOAT, 1, "tanf((float) #1)", pass_self: false
        implement Math.singleton_class, :acos, FLOAT, 1, "acosf((float) #1)", pass_self: false
    end
end

