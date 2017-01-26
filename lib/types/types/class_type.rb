# No explicit `require`s. This file should be includes via types.rb

require "set"
require_relative "../../sourcify/lib/sourcify"
require_relative "../../parsing"
require_relative "../../ast/builder"

module Ikra
    module Types
        class ClassType
            include RubyType

            attr_reader :cls
            attr_reader :inst_vars_types

            class << self
                # Ensure singleton per class
                def new(cls)
                    if @cache == nil
                        @cache = {}
                        @cache.default_proc = Proc.new do |hash, key|
                            hash[key] = super(key)
                        end
                    end

                    @cache[cls]
                end
            end

            def initialize(cls)
                @cls = cls
                @inst_vars_read = Set.new
                @inst_vars_written = Set.new

                @inst_vars_types = Hash.new
                @inst_vars_types.default_proc = Proc.new do |hash, key|
                    hash[key] = UnionType.new
                end
            end

            def inst_var_read!(inst_var_name)
                @inst_vars_read.add(inst_var_name)
            end

            def inst_var_written!(inst_var_name)
                @inst_var_written.add(inst_var_name)
            end

            def inst_var_read?(inst_var_name)
                @inst_var_read.include?(inst_var_name)
            end

            def inst_var_written(inst_var_name)
                @inst_var_written.include?(inst_var_name)
            end

            def accessed_inst_vars
                @inst_vars_read + @inst_vars_written
            end

            def to_ruby_type
                @cls
            end

            def ruby_name
                @cls.to_s
            end

            def to_c_type
                # TODO: sometimes this should be a union type struct
                "obj_id_t"
            end

            # Generates a class name for [@cls], which is a valid C++ identifier.
            #
            # For example:
            # A             --> A
            # #<Class: A>   --> singleton_A
            def class_name
                # Handle name generation for singleton classes
                return ruby_name.gsub("\#<Class:", "singleton_").gsub(">", "")
            end

            def mangled_method_name(selector)
                "_method_#{class_name}_#{selector}_"
            end

            def inst_var_array_name(inst_var_name)
                if inst_var_name.to_s[0] != "@"
                    raise "Expected instance variable identifier"
                end

                "_iv_#{class_name}_#{inst_var_name.to_s[1..-1]}_"
            end

            def method_ast(selector)
                source = Parsing.parse_method(cls.instance_method(selector))
                AST::Builder.from_parser_ast(source)
            end

            def method_parameters(selector)
                # returns names
                # TODO: handle optional params, kwargs, etc.
                to_ruby_type.instance_method(selector).parameters.map do |param|
                    param[1]
                end
            end

            def should_generate_self_arg?
                # Do not generate type for singleton classes
                return !to_ruby_type.is_a?(Module.singleton_class)
            end

            def to_s
                "<class: #{class_name}>"
            end

            def c_size
                # IDs are 4 byte integers
                4
            end
        end
    end
end

class Object
    def self.to_ikra_type
        # TODO: should this method be defined on Class?
        return Ikra::Types::ClassType.new(self)
    end

    # Returns the [Ikra::Types::RubyType] for this object. Instance of the same Ruby class can
    # principally have different Ikra types. Thus, this method is defined as an instance method.
    def ikra_type
        if self.is_a?(Module)
            return self.singleton_class.to_ikra_type
        else
            # TODO: Double check if we always want to have the singleton class?
            return self.class.to_ikra_type
        end
    end
end

