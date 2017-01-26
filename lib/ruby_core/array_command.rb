module Ikra
    module RubyIntegration
        ALL_ARRAY_COMMAND_TYPES = proc do |type|
            type.is_a?(Symbolic::ArrayCommand)
        end

        PMAP_TYPE = proc do |rcvr_type, args_ast:, block_ast:|
            # TODO: Handle keyword arguments
            rcvr_type.pmap(ast: block_ast, keep: true).to_union_type
        end

        # Implement all parallel operations
        implement ALL_ARRAY_COMMAND_TYPES, :pmap, PMAP_TYPE, 0, "<INVALID>"
    end
end

