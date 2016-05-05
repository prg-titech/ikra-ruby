require_relative "block_translator"
require_relative "command_translator"
require_relative "last_returns_visitor"
require_relative "local_variables_enumerator"
require_relative "method_translator"

module Ikra

    # This module contains functionality for translating Ruby code to CUDA (C++) code.
    module Translator
        module Constants
            ENV_IDENTIFIER = "_env_"
            ENV_DEVICE_IDENTIFIER = "dev_env"
            ENV_HOST_IDENTIFIER = "host_env"
        end

        class << self
            def wrap_in_c_block(str)
                "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
            end
        end
    end
end