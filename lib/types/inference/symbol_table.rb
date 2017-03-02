# No explicit `require`s. This file should be includes via types.rb

module Ikra
    module TypeInference
        # This is a symbol table that stores type information about variables. Ruby has a "flat" 
        # variable scope in method, i.e., variables defined in loop bodies, if statement bodies, 
        # or begin nodes are also visible outside of them.
        #
        # Every method/block has its own symbol table.
        class SymbolTable

            # Represents a lexical or local variable. Variables have a type and can read and/or
            # written, all of which is stored in this class.
            class Variable
                attr_reader :type

                # Determines the kind of the variables: lexial or local
                attr_reader :kind

                attr_accessor :read
                attr_accessor :written
                
                def initialize(type: Types::UnionType.new, kind: :local)
                    @type = type.dup
                    @kind = kind
                    @read = false
                    @written = false
                end
            end

            attr_reader :return_type

            # For debug purposes
            attr_reader :symbols

            def initialize
                # This is a mapping from variable names to Variable instances.
                @symbols = {}
                @return_type = Types::UnionType.new
            end

            def clear!
                @symbols = {}
            end

            def [](variable_name)
                if !has_variable?(variable_name)
                    raise AssertionError.new("Variable #{variable_name} not defined")
                end

                return @symbols[variable_name].type
            end

            def read!(variable_name)
                if !has_variable?(variable_name)
                    raise AssertionError.new(
                        "Variable #{variable_name} read but not found in symbol table")
                end

                @symbols[variable_name].read = true
            end

            def written!(variable_name)
                if !has_variable?(variable_name)
                    raise AssertionError.new(
                        "Variable #{variable_name} written but not found in symbol table")
                end

                @symbols[variable_name].written = true
            end

            def read_variables
                return @symbols.select do |k, v|
                    v.read
                end.keys
            end
            
            def written_variables
                return @symbols.select do |k, v|
                    v.written
                end.keys
            end
            
            def read_and_written_variables
                return read_variables + written_variables
            end

            def expand_type(variable_name, type)
                if !has_variable?(variable_name)
                    raise AssertionError.new(
                        "Attempt to expand type of variable #{variable_name} but not found in " +
                        " symbol table")
                end

                @symbols[variable_name].type.expand(type)
            end

            def expand_return_type(type)
                @return_type.expand(type)
            end

            def return_type
                return @return_type
            end

            # Declares a local variable. This method should be used only for regular local
            # variables (not parameters). Does not shadow lexical variables.
            def ensure_variable_declared(variable_name, type: Types::UnionType.new, kind: :local)
                if !has_variable?(variable_name)
                    declare_variable(variable_name, type: type, kind: kind)
                else
                    # Extend type of variable
                    expand_type(variable_name, type)
                end
            end

            # Declares a local variable and overwrites (shadows) existing variables
            # (lexical variables). Use this method for method/block parameters.
            def declare_variable(variable_name, type: Types::UnionType.new, kind: :local)
                @symbols[variable_name] = Variable.new(type: Types::UnionType.new, kind: kind)
                expand_type(variable_name, type)
            end

            private

            def has_variable?(variable_name)
                return @symbols.include?(variable_name)
            end
        end
    end
end