module Ikra
    module Translator
        class CommandTranslator < Symbolic::Visitor
            # Translate the block of an `Array.pnew` section.
            def visit_array_index_command(command)
                Log.info("Translating ArrayIndexCommand [#{command.unique_id}]")

                super

                # This is a root command, determine grid/block dimensions
                kernel_launcher.configure_grid(command.size, block_size: command.block_size)

                num_dims = command.dimensions.size

                # This is a root command, determine grid/block dimensions
                kernel_launcher.configure_grid(command.size, block_size: command.block_size)

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

                if num_dims > 1
                    # Retrieve type that was generated earlier
                    zipped_type_singleton = command.result_type.singleton_type
                    result = zipped_type_singleton.generate_inline_initialization(index_generators)

                    # Add struct type to program builder, so that we can generate the source code
                    # for its definition.
                    program_builder.structs.add(zipped_type_singleton)
                else
                    result = "_tid_"
                end

                command_translation = CommandTranslationResult.new(
                    result: result,
                    command: command)

                Log.info("DONE translating ArrayIndexCommand [#{command.unique_id}]")

                return command_translation
            end
        end
    end
end
