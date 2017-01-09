module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            def visit_array_host_section_command(command)
                Log.info("Translating ArrayHostSectionCommand [#{command.unique_id}]")

                super

                block_def_node = command.block_def_node

                # Cannot use the normal `translate_block` method here, this is special!
                # TODO: There's some duplication here with [BlockTranslator]

                # Build hash of parameter name -> type mappings
                block_parameter_types = {}
                command.block_parameter_names.each_with_index do |name, index|
                    block_parameter_types[name] = command.section_input[index].class.to_ikra_type_obj(command.section_input[index])
                end

                parameter_types_string = "[" + block_parameter_types.map do |id, type| "#{id}: #{type}" end.join(", ") + "]"
                Log.info("Translating block with input types #{parameter_types_string}")

                # Add information to block_def_node
                block_def_node.parameters_names_and_types = block_parameter_types

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                return_type = type_inference_visitor.process_block(block_def_node)

                Log.info("DONE translating ArrayHostSectionCommand [#{command.unique_id}]")
            end
        end
    end
end
