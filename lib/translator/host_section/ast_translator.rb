require_relative "../ast_translator"

module Ikra
    module Translator
        class HostSectionASTTranslator < ASTTranslator
            attr_reader :command_translator

            def initialize(command_translator:)
                super()
                @command_translator = command_translator
            end
        end
    end
end
