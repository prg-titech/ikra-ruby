require "set"
require_relative "../scope"

module Ikra
    module AST
        class MethodDefinition
            attr_accessor :type                         # receiver type
            attr_accessor :selector
            attr_accessor :ast
            attr_accessor :return_type
            attr_accessor :callers                      # method definitions calling this method
            attr_accessor :symbol_table
            attr_accessor :binding                      # needed for resolving constants
            attr_accessor :local_variables              # local variables defined in the method (name -> type)
            attr_accessor :lexical_variables            # lexical variables (defined outside; name -> type)
            attr_accessor :accessed_lexical_variables   # accessed lexical variables, only these variables are transferred to the GPU. TODO: Do we still need this? This is now determined in symbolic.rb
            attr_accessor :parameter_variables          # parameters of the method/block (name -> type)

            def initialize(type:, selector:, parameter_variables:, return_type:, ast:)
                @type = type
                @selector = selector
                @parameter_variables = parameter_variables
                @return_type = return_type
                @ast = ast
                @callers = Set.new
                @symbol_table = Scope.new
                @local_variables = {}
                @lexical_variables = {}             # optional
                @accessed_lexical_variables = {}    # optional
            end

            def parameter_names
                type.method_parameters(selector)
            end
        end
    end
end