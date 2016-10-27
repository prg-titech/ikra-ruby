module Ikra
    module RubyIntegration
        implement Math.singleton_class, :cos, FLOAT, 1, "cos((float) #1)", pass_self: false
        implement Math.singleton_class, :sin, FLOAT, 1, "sin((float) #1)", pass_self: false
        implement Math.singleton_class, :tan, FLOAT, 1, "tan((float) #1)", pass_self: false
    end
end

