module Ikra
    module RubyIntegration
        implement Math.singleton_class, :acos, FLOAT, 1, "(acosf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :acosh, FLOAT, 1, "(acoshf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :asin, FLOAT, 1, "(asinf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :asinh, FLOAT, 1, "(asinhf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :atan, FLOAT, 1, "(atanf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :atan2, FLOAT, 2, "(atan2f((float) (#1), (float) (#2)))", pass_self: false
        implement Math.singleton_class, :atanh, FLOAT, 1, "(atanhf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :cbrt, FLOAT, 1, "(cbrtf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :cos, FLOAT, 1, "(cosf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :cosh, FLOAT, 1, "(coshf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :erf, FLOAT, 1, "(erff((float) (#1)))", pass_self: false
        implement Math.singleton_class, :erfc, FLOAT, 1, "(erfcf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :exp, FLOAT, 1, "(expf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :gamma, FLOAT, 1, "(tgammaf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :hypot, FLOAT, 2, "(hypotf((float) (#1), (float) (#2)))", pass_self: false
        implement Math.singleton_class, :ldexp, FLOAT, 2, "(ldexpf((float) (#1), #2))", pass_self: false
        implement Math.singleton_class, :lgamma, FLOAT, 1, "(lgammaf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :log, FLOAT, 1, "(logf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :log10, FLOAT, 1, "(log10f((float) (#1)))", pass_self: false
        implement Math.singleton_class, :log2, FLOAT, 1, "(log2f((float) (#1)))", pass_self: false
        implement Math.singleton_class, :sin, FLOAT, 1, "(sinf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :sinh, FLOAT, 1, "(sinhf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :sqrt, FLOAT, 1, "(sqrtf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :tan, FLOAT, 1, "(tanf((float) (#1)))", pass_self: false
        implement Math.singleton_class, :tanh, FLOAT, 1, "(tanhf((float) (#1)))", pass_self: false
    end
end

