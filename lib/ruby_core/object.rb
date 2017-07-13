module Ikra
    module RubyIntegration
        OBJECT_S = Object.to_ikra_type

        implement OBJECT_S, :rand, FLOAT, 0, "curand_uniform(_env_->states + threadIdx.x + blockIdx.x * blockDim.x)", pass_self: false
    end
end
