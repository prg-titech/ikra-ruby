module Ikra
    module RubyIntegration
        MATH = Math.singleton_class.to_ikra_type

        implement MATH, :acos, FLOAT, 1, "acosf(#F0)", pass_self: false
        implement MATH, :acosh, FLOAT, 1, "acoshf(#F0)", pass_self: false
        implement MATH, :asin, FLOAT, 1, "asinf(#F0)", pass_self: false
        implement MATH, :asinh, FLOAT, 1, "asinhf(#F0)", pass_self: false
        implement MATH, :atan, FLOAT, 1, "atanf(#F0)", pass_self: false
        implement MATH, :atan2, FLOAT, 2, "atan2f(#F0, #F1)", pass_self: false
        implement MATH, :atanh, FLOAT, 1, "atanhf(#F0)", pass_self: false
        implement MATH, :cbrt, FLOAT, 1, "cbrtf(#F0)", pass_self: false
        implement MATH, :cos, FLOAT, 1, "cosf(#F0)", pass_self: false
        implement MATH, :cosh, FLOAT, 1, "coshf(#F0)", pass_self: false
        implement MATH, :erf, FLOAT, 1, "erff(#F0)", pass_self: false
        implement MATH, :erfc, FLOAT, 1, "erfcf(#F0)", pass_self: false
        implement MATH, :exp, FLOAT, 1, "expf(#F0)", pass_self: false
        implement MATH, :gamma, FLOAT, 1, "tgammaf(#F0)", pass_self: false
        implement MATH, :hypot, FLOAT, 2, "hypotf(#F0, #F1)", pass_self: false
        implement MATH, :ldexp, FLOAT, 2, "ldexpf(#F0, #F1)", pass_self: false
        implement MATH, :lgamma, FLOAT, 1, "lgammaf(#F0)", pass_self: false
        implement MATH, :log, FLOAT, 1, "logf(#F0)", pass_self: false
        implement MATH, :log10, FLOAT, 1, "log10f(#F0)", pass_self: false
        implement MATH, :log2, FLOAT, 1, "log2f(#F0)", pass_self: false
        implement MATH, :sin, FLOAT, 1, "sinf(#F0)", pass_self: false
        implement MATH, :sinh, FLOAT, 1, "sinhf(#F0)", pass_self: false
        implement MATH, :sqrt, FLOAT, 1, "sqrtf(#F0)", pass_self: false
        implement MATH, :tan, FLOAT, 1, "tanf(#F0)", pass_self: false
        implement MATH, :tanh, FLOAT, 1, "tanhf(#F0)", pass_self: false
    end
end

