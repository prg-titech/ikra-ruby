module Ikra
    module RubyIntegration
        OBJECT_S = Object.to_ikra_type

        env_identifier = Translator::Constants::ENV_IDENTIFIER
        tid = "threadIdx.x + blockIdx.x * blockDim.x"

        implement OBJECT_S, :rand, FLOAT, 0, "curand_uniform(#{env_identifier}->states + #{tid})", pass_self: false
        implement OBJECT_S, :srand, TYPE_INT_RETURN_INT, 1, "({ unsigned long long old_seed = #{env_identifier}->seeds[#{tid}]; #{env_identifier}->seeds[#{tid}] = #I0; curand_init(#I0, #{tid}, 0, _env_->states + #{tid}); old_seed; })", pass_self: false
    end
end
