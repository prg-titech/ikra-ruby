module Ikra
    module Symbolic
        class Input
            def translate_input(**kwargs)
                raise NotImplementedError.new
            end
        end

        class SingleInput < Input
            def translate_input(parent_command:, command_translator:, start_eat_params_offset: 0)
                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                parameters = [Translator::Variable.new(
                    name: parent_command.block_parameter_names[start_eat_params_offset],
                    type: input_command_translation_result.result_type)]

                return Translator::InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class ReduceInput < SingleInput
            def translate_input(parent_command:, command_translator:, start_eat_params_offset: 0)
                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                # TODO: Fix type inference (sometimes type has to be expanded)
                parameters = [
                    Translator::Variable.new(
                        name: parent_command.block_parameter_names[start_eat_params_offset],
                        type: input_command_translation_result.result_type),
                    Translator::Variable.new(
                        name: parent_command.block_parameter_names[start_eat_params_offset + 1],
                        type: input_command_translation_result.result_type)]

                return Translator::InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class StencilArrayInput < Input
            def translate_input(parent_command:, command_translator:, start_eat_params_offset: 0)
                # Parameters are allocated in a constant-sized array

                # Count number of parameters
                num_parameters = parent_command.offsets.size

                # Get single parameter name
                block_param_name = parent_command.block_parameter_names[start_eat_params_offset]

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

                return Translator::InputTranslationResult.new(
                    pre_execution: pre_execution,
                    parameters: parameters,
                    override_block_parameters: override_block_parameters,
                    command_translation_result: input_command_translation_result)
            end
        end

        class StencilSingleInput < Input
            def translate_input(parent_command:, command_translator:, start_eat_params_offset: 0)
                # Pass separate parameters

                # Translate input using visitor
                input_command_translation_result = command_translator.translate_input(self)

                # Count number of parameters
                num_parameters = parent_command.offsets.size

                # Take return type from previous computation
                parameters = []
                for index in start_eat_params_offset...(start_eat_params_offset + num_parameters)
                    parameters.push(Translator::Variable.new(
                        name: parent_command.block_parameter_names[index],
                        type: input_command_translation_result.result_type))
                end

                return Translator::InputTranslationResult.new(
                    parameters: parameters,
                    command_translation_result: input_command_translation_result)
            end
        end
    end

    module Translator
        class InputTranslationResult
            # Code to be executed before the actual execution of the block begins (but inside the
            # block function)
            attr_reader :pre_execution

            # Parameter names and types of the block (for type inference)
            attr_reader :parameters

            # Change (override) parameters of the block (to actually pass different parameters).
            # This does not affect type inference.
            attr_reader :override_block_parameters

            attr_reader :command_translation_result

            def initialize(
                pre_execution: "", 
                parameters:, 
                override_block_parameters: nil, 
                command_translation_result:)
            
                @pre_execution = pre_execution
                @parameters = parameters
                @override_block_parameters = override_block_parameters
                @command_translation_result = command_translation_result
            end
        end

        # Instance of this class store the result of translation of multiple input commands.
        # Instance methods can be used to access the values of the translated commands. Most
        # methods support access by index and access by range, in which case values are 
        # aggregated, if meaningful.
        class EntireInputTranslationResult
            def initialize(input_translation_results)
                @input = input_translation_results
            end

            def block_parameters(index = 0..-1)
                if index.is_a?(Integer)
                    return @input[index].parameters
                elsif index.is_a?(Range)
                    return @input[index].reduce([]) do |acc, n|
                        acc + n.parameters
                    end
                else
                    raise ArgumentError.new("Expected Integer or Range")
                end
            end

            def pre_execution(index = 0..-1)
                if index.is_a?(Integer)
                    return @input[index].pre_execution
                elsif index.is_a?(Range)
                    return @input[index].reduce("") do |acc, n|
                        acc + "\n" + n.pre_execution
                    end
                else
                    raise ArgumentError.new("Expected Integer or Range")
                end
            end

            def override_block_parameters(index = 0..-1)
                if index.is_a?(Integer)
                    if @input[index].override_block_parameters == nil
                        # No override specified
                        return @input[index].parameters
                    else
                        return @input[index].override_block_parameters
                    end
                elsif index.is_a?(Range)
                    return @input[index].reduce([]) do |acc, n|
                        if n.override_block_parameters == nil
                            acc + n.parameters
                        else
                            acc + n.override_block_parameters
                        end
                    end
                else
                    raise ArgumentError.new("Expected Integer or Range")
                end
            end

            def execution(index = 0..-1)
                if index.is_a?(Integer)
                     return @input[index].command_translation_result.execution
                elsif index.is_a?(Range)
                    return @input[index].reduce("") do |acc, n|
                        acc + n.command_translation_result.execution
                    end
                else
                    raise ArgumentError.new("Expected Integer or Range")
                end
            end

            def result(index = 0..-1)
                if index.is_a?(Integer)
                     return @input[index].command_translation_result.result
                elsif index.is_a?(Range)
                    return @input[index].map do |n|
                        n.command_translation_result.result
                    end
                else
                    raise ArgumentError.new("Expected Integer or Range")
                end
            end

            def command_translation_result(index)
                return @input[index].command_translation_result
            end
        end
    end
end
