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
                    hash[key] = Hash.new
                end

                @top_object_id = Hash.new
                @top_object_id.default = -1
            end

            def trace_all
                @worklist = Set.new(@roots)

                while @worklist.size > 0
                    current_obj = @worklist.first
                    @worklist.delete(current_obj)

                    trace_object(current_obj)
                end

                Log.info("Traced #{@num_traced_objects} objects with #{@objects.size} distinct types")

                @objects
            end

            def trace_object(object)
                if not object.class.to_ikra_type.is_primitive?
                    if not @objects[object.class].include_key?(object)
                        # object was not traced yet
                        @objects[object.class][object] = (@top_object_id[object.class] += 1)
                        @num_traced_objects += 1

                        object.instance_variables.each do |inst_var_name|
                            value = object.instance_variable_get(inst_var_name)
                            value_type = value.class.to_ikra_type

                            # Gather type information
                            object.class.to_ikra_type.inst_vars_types[inst_var_name].expand_with_singleton_type(value_type)

                            if value.class.include?(Entity)
                                # Keep tracing this object
                                @worklist.add(value)
                            end
                        end
                    end
                end
            end

            # Generates arrays for the Structure of Arrays (SoA) object layout
            def soa_arrays
                # arrays: class x inst var name -> Array
                arrays = Hash.new
                arrays.default_proc = do |hash, cls|
                    hash[key] = Hash.new
                    hash[key].default_proc = do |inner_hash, inst_var|
                        inner_hash[inst_var] = Array.new(@top_object_id[cls] + 1)
                    end
                end

                @objects.each do |cls, objs|
                    cls.to_ikra_type.accessed_inst_vars.each do |inst_var|
                        objs.each do |obj, id|
                            inst_var_value = obj.instance_variable_get(inst_var)

                            if !inst_var_value.class.include?(Entity)
                                Log.warn("Attempting to transfer an object that is not an Ikra::Entity. Could be a false positive. Skipping.")
                            else
                                array_value = nil
                                if inst_var_value.class.to_ikra_type.is_primitive?
                                    # Use object value directly
                                    array_value = inst_var_value
                                else
                                    # Use object ID
                                    array_value = @objects[inst_var_value.class][inst_var_value]
                                end
                                arrays[cls][inst_var][id] = array_value
                            end
                        end
                    end
                end

                arrays
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