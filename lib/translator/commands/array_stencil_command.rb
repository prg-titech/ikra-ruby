module Ikra
    def self.stencil(directions:, distance:)
        return ["G", directions, distance]
    end
    
    module Translator
        class CommandTranslator < Symbolic::Visitor

            # This visitor executes the check_parents_index function on every local variable
            # If the local variable is the "stencil array" then its indices will get modified/corrected for the 1D array access
            class FlattenIndexNodeVisitor < AST::Visitor

                attr_reader :offsets
                attr_reader :name
                attr_reader :command

                def initialize(name, offsets, command)
                    @name = name
                    @offsets = offsets
                    @command = command
                end

                def visit_lvar_read_node(node)
                    super(node)
                    check_index(name, offsets, node)
                end

                def visit_lvar_write_node(node)
                    super(node)
                    check_index(name, offsets, node)
                end


                def check_index(name, offsets, node)
                    if node.identifier == name
                        # This is the array-variable used in the stencil

                        send_node = node
                        index_combination = []
                        is_literal = true

                        # Build the index off this access in index_combination
                        for i in 0..command.dimensions.size-1
                            send_node = send_node.parent
                            if not (send_node.is_a?(AST::SendNode) && send_node.selector == :[])
                                raise AssertionError.new(
                                    "This has to be a SendNode and Array-selector")
                            end
                            index_combination[i] = send_node.arguments.first
                            if not index_combination[i].is_a?(AST::IntLiteralNode)
                                is_literal = false
                            end
                        end

                        if is_literal
                            # The index consists of only literals so we can translate it easily by mapping the index onto the offsets

                            index_combination = index_combination.map do |x|
                                x.value
                            end
                            replacement = AST::IntLiteralNode.new(value: offsets[index_combination])
                        else
                            # This handles the case where non-literals have to be translated with the Ternary Node

                            offset_arr = offsets.to_a
                            replacement = AST::IntLiteralNode.new(value: offset_arr[0][1])
                            for i in 1..offset_arr.size-1
                                # Build combination of ternary nodes

                                ternary_build = AST::SendNode.new(receiver: AST::IntLiteralNode.new(value: offset_arr[i][0][0]), selector: :==, arguments: [index_combination[0]])
                                for j in 1..index_combination.size-1

                                    next_eq = AST::SendNode.new(receiver: AST::IntLiteralNode.new(value: offset_arr[i][0][j]), selector: :==, arguments: [index_combination[j]])
                                    ternary_build = AST::SendNode.new(receiver: next_eq, selector: :"&&", arguments: [ternary_build])
                                end
                                replacement = AST::TernaryNode.new(condition: ternary_build, true_val: AST::IntLiteralNode.new(value: offset_arr[i][1]), false_val: replacement)
                            end
                        end

                        #Replace outer array access with new 1D array access

                        send_node.replace(AST::SendNode.new(receiver: node, selector: :[], arguments: [replacement]))
                    end
                end
            end

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

                # Translate relative indices to 1D-indicies starting by 0
                if command.use_parameter_array
                    offsets_mapped = Hash.new
                    for i in 0..command.offsets.size-1
                        offsets_mapped[command.offsets[i]] = i
                    end
                    command.block_def_node.accept(FlattenIndexNodeVisitor.new(command.block_parameter_names.first, offsets_mapped, command))
                end

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
                    min_in_dim = command.offsets.map do |offset|
                        offset[dim_index]
                    end.min
                    max_in_dim = command.offsets.map do |offset|
                        offset[dim_index]
                    end.max
                    
                    "temp_stencil_dim_#{dim_index} + #{min_in_dim} >= 0 && temp_stencil_dim_#{dim_index} + #{max_in_dim} < #{command.dimensions[dim_index]}"
                end.join(" && ")

                # `previous_result` should be an expression returning the array containing the
                # result of the previous computation.
                previous_result = input.result(0)

                arguments = ["_env_"]

                # Pass values from previous computation that are required by this thread.
                # Reconstruct actual indices from indices for each dimension.
                for i in 0...num_parameters
                    multiplier = 1
                    global_index = []

                    for dim_index in (num_dims - 1).downto(0)
                        global_index.push("(temp_stencil_dim_#{dim_index} + #{command.offsets[i][dim_index]}) * #{multiplier}")
                        multiplier = multiplier * command.dimensions[dim_index]
                    end

                    arguments.push("#{previous_result}[#{global_index.join(" + ")}]")
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
                    "result_type" => command.result_type.to_c_type,
                    "compute_indices" => compute_indices,
                    "out_of_bounds_check" => out_of_bounds_check,
                    "out_of_bounds_fallback" => command.out_of_range_value.to_s,
                    "stencil_computation" => stencil_computation})

                command_translation = build_command_translation_result(
                    execution: command_execution,
                    result: temp_var_name,
                    command: command)

                Log.info("DONE translating ArrayStencilCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
