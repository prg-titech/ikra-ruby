require "set"
require_relative "../entity"
require_relative "class_type"
require_relative "../symbolic/symbolic"

module Ikra
    module TypeInference

        # The object tracer determines a set of objects that are relevant for the execution of a parallel section (grouped by class). Only instances of classes that have {Ikra::Entity} included are such relevant objects.
        class ObjectTracer
            def self.process(command)
                instance = self.new(RootsFinder.process(command))
                instance.trace_all
            end

            def initialize(roots)
                # Hash map: Class -> Set[Object]
                @roots = roots
                @num_traced_objects = 0

                @objects = Hash.new
                @objects.default_proc = Proc.new do |hash, key|
                    hash[key] = Set.new
                end
            end

            def trace_all
                @worklist = Set.new(@roots)

                while @worklist.size > 0
                    current_obj = @worklist.first
                    @worklist.delete(current_obj)

                    trace_object(current_obj)
                end

                Log.info("Traced #{@num_traced_objects} objects with #{@objects.size} distinct types")

                return_value = @objects
                @objects = nil
                return_value
            end

            def trace_object(object)
                if not object.class.to_ikra_type.is_primitive?
                    if not @objects[object.class].include?(object)
                        # object was not traced yet
                        @objects[object.class].add(object)
                        @num_traced_objects += 1

                        object.instance_variables.each do |inst_var_name|
                            value = object.instance_variable_get(inst_var_name)
                            value_type = value.class.to_ikra_type

                            # Gather type information
                            object.class.to_ikra_type.inst_vars_types[inst_var_name].expand_with_singleton_type(value_type)

                            if value.class.include?(Ikra::Entity)
                                # Keep tracing this object
                                @worklist.add(value)
                            end
                        end
                    end
                end
            end

            # Finds all roots (including dependent commands) of a command.
            class RootsFinder < Symbolic::Visitor
                attr_reader :roots

                def self.process(command)
                    instance = self.new
                    command.accept(instance)
                    instance.roots
                end

                def initialize
                    @roots = Set.new
                end

                def visit_array_command(command)
                    @roots.merge(command.lexical_externals.values)
                end

                def visit_array_identity_command(command)
                    super
                    @roots.merge(command.target)
                end
            end
        end
    end
end