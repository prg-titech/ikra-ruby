require_relative "../config/configuration"

require_relative "ast_translator"
require_relative "block_translator"
require_relative "cuda_errors"
require_relative "environment_builder"
require_relative "commands/command_translator"
require_relative "last_returns_visitor"
require_relative "struct_type"
require_relative "array_command_struct_builder"

module Ikra

    # This module contains functionality for translating Ruby code to CUDA (C++) code.
    module Translator
        module Constants
            ENV_TYPE = "environment_t"
            ENV_IDENTIFIER = "_env_"
            ENV_DEVICE_IDENTIFIER = "dev_env"
            ENV_HOST_IDENTIFIER = "host_env"
            LEXICAL_VAR_PREFIX = "lex_"
            RESULT_IDENTIFIER = "_result_"
            NUM_THREADS_TYPE = "int"
            NUM_THREADS_IDENTIFIER = "_num_threads_"
            TEMP_RESULT_IDENTIFIER = "_temp_result_"
            ODD_TYPE = "bool"
            ODD_IDENTIFIER = "_odd_"
            PROGRAM_RESULT_TYPE = "result_t"
            PROGRAM_RESULT_IDENTIFIER = "program_result"
            SELF_IDENTIFIER = "_self_"

            # Make sure that these constants keep in sync with header declaration CPP file
            UNION_TYPE_SIZE = 24
            UNION_TYPE_VALUE_OFFSET = 8
        end

        class Variable
            attr_reader :type
            attr_reader :name

            def initialize(name:, type:)
                @name = name
                @type = type
            end
        end

        class << self
            def wrap_in_c_block(str)
                "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
            end

            # Reads a CUDA source code file and replaces identifiers by provided substitutes.
            # @param [String] file_name name of source code file
            # @param [Hash{String => String}] replacements replacements
            def read_file(file_name:, replacements: {})
                full_name = Ikra::Configuration.resource_file_name(file_name)
                if !File.exist?(full_name)
                    raise AssertionError.new("File does not exist: #{full_name}")
                end

                contents = File.open(full_name, "rb").read

                replacements.each do |s1, s2|
                    replacement = "/*{#{s1}}*/"
                    contents = contents.gsub(replacement, s2)
                end

                contents
            end
        end
    end
end