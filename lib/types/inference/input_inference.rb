module Ikra
    module Symbolic
        class Input
            def get_parameters(**kwargs)
                raise "Not implemented yet"
            end
        end

        class SingleInput < Input
            def get_parameters(parent_command:, start_eat_params_offset: 0)
                return [Translator::Variable.new(
                    name: parent_command.block_parameter_names[start_eat_params_offset],
                    type: command.result_type)]
            end
        end

        class ReduceInput < SingleInput
            def get_parameters(parent_command:, start_eat_params_offset: 0)
                # TODO: Fix type inference (sometimes type has to be expanded)

                return [
                    Translator::Variable.new(
                        name: parent_command.block_parameter_names[start_eat_params_offset],
                        type: command.result_type),
                    Translator::Variable.new(
                        name: parent_command.block_parameter_names[start_eat_params_offset + 1],
                        type: command.result_type)]
            end
        end

        class StencilArrayInput < Input
            def get_parameters(parent_command:, start_eat_params_offset: 0)
                # Parameters are allocated in a constant-sized array

                return [Translator::Variable.new(
                    name: parent_command.block_parameter_names[start_eat_params_offset],
                    type: command.result_type.to_array_type)]
            end
        end

        class StencilSingleInput < Input
            def get_parameters(parent_command:, start_eat_params_offset: 0)
                # Pass separate parameters

                # Count number of parameters
                num_parameters = parent_command.offsets.size

                # Take return type from previous computation
                parameters = []
                for index in start_eat_params_offset...(start_eat_params_offset + num_parameters)
                    parameters.push(Translator::Variable.new(
                        name: parent_command.block_parameter_names[index],
                        type: command.result_type))
                end

                return parameters
            end
        end
    end
    

end
