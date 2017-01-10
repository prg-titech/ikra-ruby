require "set"
require "ffi"
require_relative "../../entity"
require_relative "../../symbolic/symbolic"
require_relative "../../translator/commands/command_translator"

module Ikra
    module TypeInference

        # The object tracer determines a set of objects that are relevant for the execution of a parallel section (grouped by class). Only instances of classes that have {Ikra::Entity} included are such relevant objects.
        class ObjectTracer
            def initialize(command)
                # Hash map: Class -> Set[Object]
                @roots = RootsFinder.process(command)
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
                    if not @objects[object.class].has_key?(object)
                        # object was not traced yet
                        @objects[object.class][object] = (@top_object_id[object.class] += 1)
                        @num_traced_objects += 1

                        object.instance_variables.each do |inst_var_name|
                            value = object.instance_variable_get(inst_var_name)
                            value_type = value.class.to_ikra_type_obj(value)

                            # Gather type information
                            object.class.to_ikra_type_obj(object).inst_vars_types[inst_var_name].add(value_type)

                            if value.class.include?(Entity)
                                # Keep tracing this object
                                @worklist.add(value)
                            end
                        end
                    end
                end
            end

            # Generates arrays for the Structure of Arrays (SoA) object layout
            def register_soa_arrays(environment_builder)
                # arrays: class x inst var name -> Array
                arrays = Hash.new
                arrays.default_proc = proc do |hash, cls|
                    inner_hash = Hash.new
                    inner_hash.default_proc = proc do |inner, inst_var|
                        inner[inst_var] = Array.new(@top_object_id[cls] + 1)
                    end
                    hash[cls] = inner_hash
                end

                @objects.each do |cls, objs|
                    cls.to_ikra_type.accessed_inst_vars.each do |inst_var|
                        objs.each do |obj, id|
                            inst_var_value = obj.instance_variable_get(inst_var)

                            if inst_var_value.class.to_ikra_type.is_primitive?
                                # Use object value directly
                                arrays[cls][inst_var][id] = inst_var_value
                            else
                                if !inst_var_value.class.include?(Entity)
                                    Log.warn("Attempting to transfer an object of class #{inst_var_value.class} that is not an Ikra::Entity. Could be a false positive. Skipping.")
                                else
                                    # Use object ID
                                    arrays[cls][inst_var][id] = @objects[inst_var_value.class][inst_var_value]
                                end
                            end
                        end
                    end
                end
                
                arrays.each do |cls, inner_hash|
                    inner_hash.each do |inst_var, array|
                        environment_builder.add_soa_array(cls.to_ikra_type.inst_var_array_name(inst_var), array)
                    end
                end
            end

            # Returns an array of IDs for the base array or the base array itself if all values are primitive. 
            # TODO: This should be done in the environment builder, but I want to avoid copying over arrays...
            def convert_base_array(base_array, need_union_type)
                if base_array.first.class.to_ikra_type.is_primitive? && !need_union_type
                    base_array
                else
                    if !need_union_type
                        base_array.map do |obj|
                            @objects[obj.class][obj]
                        end
                    else
                        mem_block = FFI::MemoryPointer.new(Translator::EnvironmentBuilder::UnionTypeStruct, base_array.size)
                        array = base_array.size.times.collect do |index|
                            Translator::EnvironmentBuilder::UnionTypeStruct.new(mem_block + index * Translator::EnvironmentBuilder::UnionTypeStruct.size)
                        end

                        base_array.each_with_index do |obj, index|
                            obj_type = obj.class.to_ikra_type
                            array[index][:class_id] = obj_type.class_id

                            if obj_type.is_primitive?
                                # TODO: what if the primitive value is not an integer?
                                array[index][:object_id] = obj
                            else
                                array[index][:object_id] = @objects[obj.class][obj]
                            end
                        end

                        mem_block
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