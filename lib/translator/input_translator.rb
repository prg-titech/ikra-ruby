module Ikra
    module Symbolic
        class Input
            def translate_input(**kwargs)
                raise "Not implemented yet"
            end
        end

        class SingleInput < Input
            def translate_input(command:, command_translator:, start_eat_params_offset: 0)
                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                parameters = [Translator::Variable.new(
                    name: command.block_parameter_names[start_eat_params_offset],
                    type: input_command_translation_result.result_type)]

                return InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class ReduceInput < SingleInput
            def translate_input(command:, command_translator:, start_eat_params_offset: 0)
                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                # TODO: Fix type inference (sometimes type has to be expanded)
                parameters = [
                    Translator::Variable.new(
                        name: command.block_parameter_names[start_eat_params_offset],
                        type: input_command_translation_result.result_type),
                    Translator::Variable.new(
                        name: command.block_parameter_names[start_eat_params_offset + 1],
                        type: input_command_translation_result.result_type)]

                return InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class StencilArrayInput < Input
            def translate_input(command:, command_translator:, start_eat_params_offset: 0)
                # Parameters are allocated in a constant-sized array

                # Count number of parameters
                num_parameters = command.offsets.size

                # Get single parameter name
                block_param_name = command.block_parameter_names[start_eat_params_offset]

                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                # Take return type from previous computation
                parameters = [Translator::Variable.new(
                    name: block_param_name,
                    type: input_command_translation_result.result_type.to_array_type)]


                # Allocate and fill array of parameters
                actual_parameter_names = (0...num_parameters).map do |param_index| 
                    "_#{block_param_name}_#{param_index}"
                end

                param_array_init = "{ " + actual_parameter_names.join(", ") + " }"

                pre_execution = Translator.read_file(file_name: "stencil_array_reconstruction.cpp", replacements: {
                    "type" => input_command_translation_result.result_type.to_c_type,
                    "name" => block_param_name.to_s,
                    "initializer" => param_array_init})

                # Pass multiple single values instead of array
                override_block_parameters  = actual_parameter_names.map do |param_name|
                    Translator::Variable.new(
                        name: param_name,
                        type: input_command_translation_result.result_type)
                end

                return InputTranslationResult.new(
                    pre_execution: pre_execution,
                    parameters: parameters,
                    override_block_parameters: override_block_parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class StencilSingleInput < Input
            def translate_input(command:, command_translator:, start_eat_params_offset: 0)
                # Pass separate parameters

                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                # Count number of parameters
                num_parameters = command.offsets.size

                # Take return type from previous computation
                parameters = []
                for index in start_eat_params_offset...(start_eat_params_offset + num_parameters)
                    parameters.push(Translator::Variable.new(
                        name: command.block_parameter_names[index],
                        type: input_command_translation_result.result_type))
                end

                return InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class InputTranslationResult
            # Code to be executed before the actual execution of the block begins (but inside the
            # block function)
            attr_reader :pre_execution

            # Parameter names and types of the block (for type inference)
            attr_reader :parameters

            # Change (override) parameters of the block (to actually pass different parameters)
            attr_reader :override_block_parameters

            attr_reader :command_translation_result

            def initialize(pre_execution: "", parameters:, override_block_parameters: nil, command_translation_result:)
                @pre_execution = pre_execution
                @parameters = parameters
                @override_block_parameters = override_block_parameters
                @command_translation_result = command_translation_result
            end

            def result_type
                return command_translation_result.result_type
            end
        end
    end
end