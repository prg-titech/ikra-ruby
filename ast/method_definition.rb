module Ikra
    module AST
        class MethodDefinition
            attr_accessor :type
            attr_accessor :selector
            attr_accessor :ast
            attr_accessor :parameter_types
            attr_accessor :return_type

            def initialize(type:, selector:, parameter_types:, return_type:, ast:)
                @type = type
                @selector = selector
                @parameter_types = parameter_types
                @return_type = return_type
                @ast = ast
            end
        end
    end
end