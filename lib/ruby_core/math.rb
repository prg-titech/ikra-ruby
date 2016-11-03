module Ikra
    module RubyIntegration
        MATH = Math.singleton_class.to_ikra_type

        implement MATH, :acos, FLOAT, 1, "(acosf((float) (#1)))", pass_self: false
        implement MATH, :acosh, FLOAT, 1, "(acoshf((float) (#1)))", pass_self: false
        implement MATH, :asin, FLOAT, 1, "(asinf((float) (#1)))", pass_self: false
        implement MATH, :asinh, FLOAT, 1, "(asinhf((float) (#1)))", pass_self: false
        implement MATH, :atan, FLOAT, 1, "(atanf((float) (#1)))", pass_self: false
        implement MATH, :atan2, FLOAT, 2, "(atan2f((float) (#1), (float) (#2)))", pass_self: false
        implement MATH, :atanh, FLOAT, 1, "(atanhf((float) (#1)))", pass_self: false
        implement MATH, :cbrt, FLOAT, 1, "(cbrtf((float) (#1)))", pass_self: false
        implement MATH, :cos, FLOAT, 1, "(cosf((float) (#1)))", pass_self: false
        implement MATH, :cosh, FLOAT, 1, "(coshf((float) (#1)))", pass_self: false
        implement MATH, :erf, FLOAT, 1, "(erff((float) (#1)))", pass_self: false
        implement MATH, :erfc, FLOAT, 1, "(erfcf((float) (#1)))", pass_self: false
        implement MATH, :exp, FLOAT, 1, "(expf((float) (#1)))", pass_self: false
        implement MATH, :gamma, FLOAT, 1, "(tgammaf((float) (#1)))", pass_self: false
        implement MATH, :hypot, FLOAT, 2, "(hypotf((float) (#1), (float) (#2)))", pass_self: false
        implement MATH, :ldexp, FLOAT, 2, "(ldexpf((float) (#1), #2))", pass_self: false
        implement MATH, :lgamma, FLOAT, 1, "(lgammaf((float) (#1)))", pass_self: false
        implement MATH, :log, FLOAT, 1, "(logf((float) (#1)))", pass_self: false
        implement MATH, :log10, FLOAT, 1, "(log10f((float) (#1)))", pass_self: false
        implement MATH, :log2, FLOAT, 1, "(log2f((float) (#1)))", pass_self: false
        implement MATH, :sin, FLOAT, 1, "(sinf((float) (#1)))", pass_self: false
        implement MATH, :sinh, FLOAT, 1, "(sinhf((float) (#1)))", pass_self: false
        implement MATH, :sqrt, FLOAT, 1, "(sqrtf((float) (#1)))", pass_self: false
        implement MATH, :tan, FLOAT, 1, "(tanf((float) (#1)))", pass_self: false
        implement MATH, :tanh, FLOAT, 1, "(tanhf((float) (#1)))", pass_self: false
    end
end

