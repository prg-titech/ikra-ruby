require "set"

module Ikra
    module AST
        # Converts an AST to single static assignment form (SSA). 
        # TODO: Make SSA form minimal, i.e., remove unnecessary assignments (Phi node artifacts).
        class SSAGenerator < Visitor
            def self.transform_to_ssa!(node)
                node.accept(self.new)
            end

            def initialize
                @aliases = {}
                @ssa_id = 0
            end

            def new_ssa_var_name(old_name = nil)
                @ssa_id = @ssa_id + 1

                if old_name == nil
                    return "_ssa_var_#{@ssa_id}"
                else
                    return "_ssa_var_#{old_name}_#{@ssa_id}"
                end
            end

            def visit_lvar_read_node(node)
                super

                if @aliases.include?(node.identifier)
                    # This variable was renamed
                    node.replace(LVarReadNode.new(identifier: @aliases[node.identifier]))
                end
            end

            def visit_lvar_write_node(node)
                super

                # Give the variable a new name
                new_name = new_ssa_var_name(node.identifier)
                @aliases[node.identifier] = new_name

                node.replace(LVarWriteNode.new(
                    identifier: new_name,
                    value: node.value))
            end

            # Merges all "alias" hash maps given by `branch_aliases`. If a conflict is detected,
            # i.e., a variable is renamed to different names in at least to branches, `block` is
            # executed with the original variable name as argument. The block should return the
            # new resolved variable name.
            def merge_aliases(*branch_aliases, &block)
                result = {}

                orig_names = Set.new(branch_aliases.map do |aliases|
                    aliases.keys
                end.flatten)

                for orig_name in orig_names
                    new_names = Set.new(branch_aliases.map do |aliases|
                        aliases[orig_name]
                    end)

                    if new_names.size > 1
                        # Renamed to different values
                        result[orig_name] = yield(orig_name)
                    else
                        result[orig_name] = new_names.first
                    end
                end

                return result
            end

            def visit_if_node(node)
                node.condition.accept(self)

                branch1_aliases = @aliases.dup
                branch2_aliases = @aliases.dup

                @aliases = branch1_aliases
                node.true_body_stmts.accept(self)
                branch1_aliases = @aliases

                @aliases = branch2_aliases
                node.false_body_stmts.accept(self)
                branch2_aliases = @aliases

                @aliases = merge_aliases(branch1_aliases, branch2_aliases) do |orig_var|
                    # Conflict found: `orig_var` renamed in both branches
                    name1 = branch1_aliases[orig_var]
                    name2 = branch2_aliases[orig_var]

                    resolved_name = new_ssa_var_name(orig_var)

                    if not node.true_body_stmts.is_a?(BeginNode)
                        raise AssertionError.new("Expected a BeginNode")
                    end

                    if not node.false_body_stmts.is_a?(BeginNode)
                        raise AssertionError.new("Expected a BeginNode")
                    end

                    # TODO: Should check liveness of variables
                    node.true_body_stmts.add_statement(LVarWriteNode.new(
                        identifier: resolved_name, value: LVarReadNode.new(identifier: name1)))
                    node.false_body_stmts.add_statement(LVarWriteNode.new(
                        identifier: resolved_name, value: LVarReadNode.new(identifier: name2)))

                    resolved_name
                end
            end

            # TODO: Handle `TernaryNode`

            def visit_for_node(node)
                node.range_from.accept(self)
                node.range_to.accept(self)

                before_loop_aliases = @aliases.dup
                loop_body_aliases = @aliases.dup

                @aliases = loop_body_aliases
                node.body_stmts.accept(self)

                # Insert merge statements at the end of the loop body
                @aliases = merge_aliases(before_loop_aliases, loop_body_aliases) do |orig_var|
                    # Conflict found
                    name_before = before_loop_aliases[orig_var]
                    name_inside = loop_body_aliases[orig_var]

                    node.body_stmts.add_statement(LVarWriteNode.new(
                        identifier: name_before, value: LVarReadNode.new(identifier: name_inside)))

                    # Resolved name is `name_before`
                    name_before
                end
            end

            def process_while_until_node(node)
                node.condition.accept(self)

                before_loop_aliases = @aliases.dup
                loop_body_aliases = @aliases.dup

                @aliases = loop_body_aliases
                node.body_stmts.accept(self)

                # Insert merge statements at the end of the loop body
                @aliases = merge_aliases(before_loop_aliases, loop_body_aliases) do |orig_var|
                    # Conflict found
                    name_before = before_loop_aliases[orig_var]
                    name_inside = loop_body_aliases[orig_var]

                    node.body_stmts.add_statement(LVarWriteNode.new(
                        identifier: name_before, value: LVarReadNode.new(identifier: name_inside)))

                    # Resolved name is `name_before`
                    name_before
                end
            end

            def visit_while_node(node)
                process_while_until_node(node)
            end

            def visit_until_node(node)
                process_while_until_node(node)
            end

            def visit_while_post_node(node)
                process_while_until_node(node)
            end

            def visit_until_post_node(node)
                process_while_until_node(node)
            end
        end
    end
end