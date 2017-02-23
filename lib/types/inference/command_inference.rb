module Ikra
    module TypeInference
        class CommandInference < Symbolic::Visitor
            def self.process_command(command)
                return command.accept(CommandInference.new)
            end

            # Processes a parallel section, i.e., a [BlockDefNode]. Performs the following steps:
            # 1. Gather parameter types and names in a hash map.
            # 2. Set lexical variables on [BlockDefNode].
            # 3. Perform type inference, i.e., annotate [BlockDefNode] AST with types.
            # 4. Return result type of the block.
            def process_block(
                block_def_node:, 
                lexical_variables: {}, 
                block_parameters:)

                # Build hash of parameter name -> type mappings
                block_parameter_types = {}
                for variable in block_parameters
                    block_parameter_types[variable.name] = variable.type
                end

                parameter_types_string = "[" + block_parameter_types.map do |id, type| "#{id}: #{type}" end.join(", ") + "]"
                Log.info("Type inference for block with input types #{parameter_types_string}")

                # Add information to block_def_node
                block_def_node.parameters_names_and_types = block_parameter_types

                # Lexical variables
                lexical_variables.each do |name, value|
                    block_def_node.lexical_variables_names_and_types[name] = value.ikra_type.to_union_type
                end

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                return_type = type_inference_visitor.process_block(block_def_node)

                return return_type
            end

            # Processes all input commands. This is similar to `translate_entire_input`, but it
            # performs only type inference and does not generate any source code.
            def process_entire_input(command)
                input_parameters = command.input.each_with_index.map do |input, index|
                    input.get_parameters(
                        parent_command: command,
                        # Assuming that every input consumes exactly one parameter
                        start_eat_params_offset: index)
                end

                return input_parameters.reduce(:+)
            end

            # Process block and dependent computations. This method is used for all array
            # commands that do not have a separate Visitor method.
            def visit_array_command(command)
                return process_block(
                    block_def_node: command.block_def_node,
                    lexical_variables: command.lexical_externals,
                    block_parameters: process_entire_input(command))
            end

            def visit_array_in_host_section_command(command)
                return command.base_type
            end

            def visit_array_identity_command(command)
                return command.base_type
            end

            def visit_array_index_command(command)
                num_dims = command.dimensions.size

                if num_dims > 1
                    # Build Ikra struct type
                    zipped_type_singleton = Types::ZipStructType.new(
                        *([Types::UnionType.create_int] * command.dimensions.size))
                    return zipped_type_singleton.to_union_type
                else
                    return Types::UnionType.create_int
                end
            end

            def visit_array_zip_command(command)
                input_types = command.input.each_with_index.map do |input, index|
                    input.get_parameters(
                        parent_command: command,
                        # Assuming that every input consumes exactly one parameter
                        start_eat_params_offset: index).map do |variable|
                            variable.type
                        end
                end.reduce(:+)

                # Build Ikra struct type
                zipped_type_singleton = Types::ZipStructType.new(*input_types)
                return zipped_type_singleton.to_union_type
            end
        end
    end
end