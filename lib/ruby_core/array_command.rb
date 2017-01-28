module Ikra
    module RubyIntegration
        ALL_ARRAY_COMMAND_TYPES = proc do |type|
            type.is_a?(Symbolic::ArrayCommand)
        end

        PMAP_TYPE = proc do |rcvr_type, args_ast:, block_ast:|
            # TODO: Handle keyword arguments
            rcvr_type.pmap(ast: block_ast, keep: true).to_union_type
        end

        COMMAND_INNER_TYPE = proc do |rcvr_type, args_ast:, block_ast:|
            # TODO: Handle keyword arguments
            rcvr_type.result_type.to_array_type.to_union_type
        end

        LAUNCH_KERNEL = proc do

        end

        # Implement all parallel operations
        implement ALL_ARRAY_COMMAND_TYPES, :pmap, PMAP_TYPE, 0, "new array_command()"
        implement ALL_ARRAY_COMMAND_TYPES, :__call__, COMMAND_INNER_TYPE, 0, LAUNCH_KERNEL
    end
end

