require "set"
require_relative "../scope"

module Ikra
    module AST
        class MethodDefinition
            attr_accessor :type
            attr_accessor :selector
            attr_accessor :ast
            attr_accessor :return_type
            attr_accessor :callers
            attr_accessor :symbol_table
            attr_accessor :binding
            attr_accessor :local_variables
            attr_accessor :lexical_variables
            attr_accessor :parameter_variables

            def initialize(type:, selector:, parameter_variables:, return_type:, ast:)
                @type = type
                @selector = selector
                @parameter_variables = parameter_variables
                @return_type = return_type
                @ast = ast
                @callers = Set.new
                @symbol_table = Scope.new
                @local_variables = {}
                @lexical_variables = {}     # optional
            end

            def parameter_names
                type.method_parameters(selector)
            end
        end
    end
end