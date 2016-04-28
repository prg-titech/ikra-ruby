require_relative "translator"

module Ikra
    module Translator

        # Result of translating a [Symbolic::ArrayCommand].
        class CommandTranslationResult
            attr_accessor :environment_builder
            attr_accessor :generated_source
            attr_accessor :invocation

            def initialize
                @environment_builder = EnvironmentBuilder.new           # [EnvironmentBuilder] instance that generates the struct containing accessed lexical variables.
                @generated_source = ""                                  # [String] containing the currently generated source code.
                @invocation = "NULL"                                    # [String] source code used for invoking the block function.
                @return_type = Types::UnionType.new                     # [Types::UnionType] return type of the block.
            end

            def clone
                # TODO: do we really need this?
                cloned = super
                @cloned.environment_builder = @environment_builder.clone
                cloned
            end
        end

        class CommandFFIWrapper

        end

        class ArrayCommandVisitor < Symbolic::Visitor
            def visit_array_new_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    # only one block parameter (int)
                    block_parameter_types: {command.block_parameter_names.first => Types::UnionType.create_int},
                    env_builder: command_translation_result.environment_builder,
                    lexical_variables: command.lexical_externals)

                command_translation_result.generated_source = block_translation_result.c_source

                # CONTINUE HERE... TO DO: generate source for aux_methods, ....
            end

            def visit_array_identity_command(command)
                # create brand new result
                result = CommandTranslationResult.new
                result.invocation = 
            end

            def visit_array_map_command(command)
                dependent_result = super.clone
            end
        end

        class << self
            def translate_command(array_command)

            end
        end
    end
end