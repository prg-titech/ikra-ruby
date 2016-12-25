module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_stencil_command(command)
                Log.info("Translating ArrayStencilCommand [#{command.unique_id}]")

                super

                num_dims = command.dimensions.size

                # Process dependent computation (receiver), returns [InputTranslationResult]
                input = translate_entire_input(command)

                # Count number of parameters
                num_parameters = command.offsets.size

                # All variables accessed by this block should be prefixed with the unique ID
                # of the command in the environment.
                env_builder = @environment_builder[command.unique_id]

                block_translation_result = Translator.translate_block(
                    block_def_node: command.block_def_node,
                    environment_builder: env_builder,
                    lexical_variables: command.lexical_externals,
                    command_id: command.unique_id,
                    entire_input_translation: input)

                kernel_builder.add_methods(block_translation_result.aux_methods)
                kernel_builder.add_block(block_translation_result.block_source)

                # Compute indices in all dimensions
                index_generators = (0...num_dims).map do |dim_index|
                    index_div = command.dimensions.drop(dim_index + 1).reduce(1, :*)
                    index_mod = command.dimensions[dim_index]

                    if dim_index > 0
                         "(_tid_ / #{index_div}) % #{index_mod}"
                    else
                        # No modulo required for first dimension
                        "_tid_ / #{index_div}"
                    end
                end

                compute_indices = index_generators.map.with_index do |gen, dim_index|
                    "int temp_stencil_dim_#{dim_index} = #{gen};"
                end.join("\n")

                # Check if an index is out of bounds in any dimension
                out_of_bounds_check = Array.new(num_dims) do |dim_index|
                    if num_dims > 1
                        min_in_dim = command.offsets.map do |offset|
                            offset[dim_index]
                        end.min
                        max_in_dim = command.offsets.map do |offset|
                            offset[dim_index]
                        end.max
                    else
                        min_in_dim = command.offsets.min
                        max_in_dim = command.offsets.max
                    end

                    "temp_stencil_dim_#{dim_index} + #{min_in_dim} >= 0 && temp_stencil_dim_#{dim_index} + #{max_in_dim} < #{command.dimensions[dim_index]}"
                end.join(" && ")

                # `previous_result` should be an expression returning the array containing the
                # result of the previous computation.
                previous_result = input.result(0)

                arguments = ["_env_"]

                # Pass values from previous computation that are required by this thread.
                # Reconstruct actual indices from indices for each dimension.
                for i in 0...num_parameters
                    if num_dims > 1
                        multiplier = 1
                        global_index = []

                        for dim_index in (num_dims - 1).downto(0)
                            global_index.push("(temp_stencil_dim_#{dim_index} + #{command.offsets[i][dim_index]}) * #{multiplier}")
                            multiplier = multiplier * command.dimensions[dim_index]
                        end

                        arguments.push("#{previous_result}[#{global_index.join(" + ")}]")
                    else
                        arguments.push("#{previous_result}[temp_stencil_dim_0 + #{command.offsets[i]}]")
                    end
                end

                # Push additional arguments (e.g., index)
                arguments.push(*input.result(1..-1))
                argument_str = arguments.join(", ")
                stencil_computation = block_translation_result.function_name + "(#{argument_str})"

                temp_var_name = "temp_stencil_#{CommandTranslator.next_unique_id}"

                # The following template checks if there is at least one index out of bounds. If
                # so, the fallback value is used. Otherwise, the block is executed.
                command_execution = Translator.read_file(file_name: "stencil_body.cpp", replacements: {
                    "execution" => input.execution,
                    "temp_var" => temp_var_name,
                    "result_type" => block_translation_result.result_type.to_c_type,
                    "compute_indices" => compute_indices,
                    "out_of_bounds_check" => out_of_bounds_check,
                    "out_of_bounds_fallback" => command.out_of_range_value.to_s,
                    "stencil_computation" => stencil_computation})

                command_translation = build_command_translation_result(
                    execution: command_execution,
                    result: temp_var_name,
                    result_type: block_translation_result.result_type,
                    keep: command.keep,
                    unique_id: command.unique_id,
                    command: command)

                kernel_launcher.set_result_name(command.unique_id.to_s)

                Log.info("DONE translating ArrayStencilCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
