module Ikra
    module Translator
        # This class looks for `input_command` in the tree of commands (and inputs) provided
        # by `relative_command`. It traverses this tree and generates and expression that can
        # be used to access `input_command` from `relative_command`, which is represented by
        # an array_command struct in the CUDA code and referenced with `command_expr`.
        # TODO: This class should be a visitor, but we need to pass additional values (`path`)
        # along the way.
        module KernelLaunchArgumentGenerator
            def self.generate_arg(input_command, relative_command, command_expr)
                return visit_array_command(input_command, relative_command, command_expr)
            end

            def self.visit_array_command(input_command, command, path)
                if command.is_a?(Symbolic::ArrayInHostSectionCommand)
                    if command.equal?(input_command)
                        # This should be passed as an argument
                        return "((#{command.base_type.to_c_type} *) #{path}->input_0.content)"
                    else
                        # This is not the one we are looking for
                        return nil
                    end
                else
                    command.input.each_with_index do |input, index|
                        if input.command.is_a?(Symbolic::ArrayCommand)
                            result = visit_array_command(
                                input_command, input.command, "#{path}->input_#{index}")

                            if result != nil
                                return "((#{input_command.base_type.to_c_type} *) #{result})"
                            end
                        end
                    end

                    return nil
                end
            end
        end

        class ArrayCommandStructBuilder < Symbolic::Visitor
            def self.struct_name(command)
                return "array_command_#{command.unique_id}"
            end

            # This class determines if a `size` instance method should be generated for an
            # array_command struct type. This is the case iff struct, or its first input, or
            # the first input of its first input, etc., is an ArrayInHostSectionCommand.
            # The size of such arrays is in general not known at compile time.
            class RequireRuntimeSizeChecker < Symbolic::Visitor
                def self.require_size_function?(command)
                    return command.accept(self.new)
                end

                def visit_array_reduce_command(command)
                    # Size is always 1
                    return false
                end

                def visit_array_identity_command(command)
                    # Fully fused, size known at compile time
                    return false
                end

                def visit_array_in_host_section_command(command)
                    # Cannot be fused, size unknown at compile time
                    return true
                end

                def visit_fixed_size_array_in_host_section_command(command)
                    # Size is part of the type/command
                    return false
                end

                def visit_array_command(command)
                    if command.input.size == 0
                        return false
                    else
                        return command.input.first.command.accept(self)
                    end
                end
            end

            # This class builds a struct containing references to input (depending) commands for
            # a certain array command. It is a subclass of [Symbolic::Visitor] but does not 
            # traverse the tree. We just take advantage of the double dispatch here.
            class SingleStructBuilder < Symbolic::Visitor
                def struct_name(command)
                    return ArrayCommandStructBuilder.struct_name(command)
                end

                def visit_array_command(command)
                    this_name = struct_name(command)
                    struct_def = "struct #{this_name} {\n"
                    
                    # Debug information
                    struct_def = struct_def + "    // #{command.class}\n"

                    # Generate fields
                    struct_def = struct_def + "    #{command.result_type.to_c_type} *result;\n"

                    all_params = ["#{command.result_type.to_c_type} *result = NULL"]
                    all_initializers = ["result(result)"]

                    command.input.each_with_index do |input, index|
                        if input.command.is_a?(Symbolic::ArrayCommand)
                            struct_def = struct_def + "    #{struct_name(input.command)} *input_#{index};\n"
                            all_params.push("#{struct_name(input.command)} *input_#{index} = NULL")
                            all_initializers.push("input_#{index}(input_#{index})")
                        end
                    end

                    # Generate constructor
                    struct_def = struct_def + "    __host__ __device__ #{this_name}(#{all_params.join(', ')}) : #{all_initializers.join(', ')} { }\n"

                    # Add instance methods
                    if RequireRuntimeSizeChecker.require_size_function?(command)
                        # ArrayIndexCommand does not have any input, as an example. But in this
                        # case, we also do not need the `size` function, because it is a root
                        # command that can be fused.
                        struct_def = struct_def + "    int size() { return input_0->size(); }\n"
                    end

                    struct_def = struct_def + "};"
                end

                def visit_array_in_host_section_command(command)
                    this_name = struct_name(command)
                    struct_def = "struct #{this_name} {\n"

                    # Debug information
                    struct_def = struct_def + "    // #{command.class}\n"
                    
                    struct_def = struct_def + "    #{command.result_type.to_c_type} *result;\n"
                    struct_def = struct_def + "    variable_size_array_t input_0;\n"
                    struct_def = struct_def + "    __host__ __device__ #{this_name}(#{command.result_type.to_c_type} *result = NULL, variable_size_array_t input_0 = variable_size_array_t::error_return_value) : result(result), input_0(input_0) { }\n"

                    # Add instance methods
                    struct_def = struct_def + "    int size() { return input_0.size; }\n"

                    return struct_def + "};"
                end
            end

            attr_reader :all_structs

            def initialize
                @all_structs = []
                @builder = SingleStructBuilder.new
            end

            def visit_array_command(command)
                super
                @all_structs.push(command.accept(@builder))
            end

            def self.build_all_structs(command)
                visitor = self.new
                command.accept(visitor)
                return visitor.all_structs
            end
        end
    end
end
