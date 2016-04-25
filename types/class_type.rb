require "set"
require_relative "ruby_type"
require_relative "../sourcify/lib/sourcify"
require_relative "../parsing"
require_relative "../ast/builder"

module Ikra
    module Types
        class ClassType
            include RubyType

            attr_reader :cls

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

            def to_ruby_type
                @cls
            end

            def ruby_name
                @cls.to_s
            end

            def to_c_type
                # TODO: sometimes this should be a union type struct
                "objid_t"
            end

            def mangled_method_name(selector)
                "_method_#{@cls.to_s}_#{selector}_"
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

            def should_generate_type?
                to_ruby_type != Class
            end

            def to_s
                "<class: #{@cls.to_s}>"
            end
        end
    end
end

class Object
    def self.to_ikra_type
        Ikra::Types::ClassType.new(self)
    end
end