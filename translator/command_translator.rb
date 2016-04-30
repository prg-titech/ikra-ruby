require_relative "translator"

module Ikra
    module Translator

        # Result of translating a [Symbolic::ArrayCommand].
        class CommandTranslationResult
            attr_accessor :environment_builder
            attr_accessor :generated_source
            attr_accessor :invocation
            attr_accessor :size
            attr_accessor :return_type

            def initialize
                @environment_builder = EnvironmentBuilder.new           # [EnvironmentBuilder] instance that generates the struct containing accessed lexical variables.
                @generated_source = ""                                  # [String] containing the currently generated source code.
                @invocation = "NULL"                                    # [String] source code used for invoking the block function.
                @return_type = Types::UnionType.new                     # [Types::UnionType] return type of the block.
                @size = 0                                               # [Fixnum] number of elements in base array
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

                command_translation_result.generated_source = block_translation_result.generated_source

                tid = "threadIdx.x + blockIdx.x * blockDim.x"
                command_translation_result.invocation = "#{block_translation_result.function_name}(#{EnvParameterName}, #{tid})"
                command_translation_result.size = command.size
                command_translation_result.return_type = block_translation_result.result_type
            end

            def visit_array_identity_command(command)
                # create brand new result
                command_translation_result = CommandTranslationResult.new

                # no source code generation
                command_translation_result.invocation = "_input_k#{command.unique_id}_[threadIdx.x + blockIdx.x * blockDim.x]"
                command_translation_result.size = command.size
                command_translation_result.return_type = command.base_type
            end

            def visit_array_map_command(command)
                dependent_result = super                            # visit target (dependent) command
                command_translation_result = CommandTranslationResult.new
                command_translation_result.environment_builder = dependent_result.environment_builder.clone

                block_translation_result = Translator.translate_block(
                    ast: command.ast,
                    block_parameter_types: {command.block_parameter_names.first => dependent_result.return_type},
                    env_builder: command_translation_result.environment_builder,
                    lexical_variables: command.lexical_variables)

                command_translation_result.generated_source = dependent_result.generated_source + "\n\n" + block_translation_result.generated_source

                command_translation_result.invocation = "#{block_translation_result.function_name}(#{EnvParameterName}, #{dependent_result.invocation})"
                command_translation_result.size = dependent_result.size
                command_translation_result.return_type = block_translation_result.result_type
            end
        end

        class << self
            def translate_command(array_command)

            end
        end
    end
end