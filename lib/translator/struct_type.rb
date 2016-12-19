module Ikra
    module Types
        class StructType
            def generate_definition
                raise NotImplementedError.new("ZipStructType is the only implementation")
            end
        end

        class ZipStructType < StructType
            # Generates a source code expression that creates and initializes an instance of
            # this struct.
            def generate_inline_initialization(*input)
                field_init = input.join(", ")
                return "((#{to_c_type}) {#{field_init}})"
            end

            def generate_definition
                fields_def = @fields.map do |field_name, type|
                    "#{type.to_c_type} #{field_name};"
                end

                all_fields = fields_def.join("\n")

                return Translator.read_file(file_name: "struct_definition.cpp", replacements: {
                    "name" => to_c_type,
                    "fields" => all_fields})
            end

            # Generates a source code expression that reads a fields of this struct by index.
            def generate_read(receiver, selector, index)
                # Type inference already ensured that there is exactly one parameter which is
                # an IntLiteral.

                return "#{receiver}.field_#{index}"
            end

            def generate_non_constant_read(receiver, selector, index_expression_identifier)
                expression = ""

                for index in 0...@fields.size
                    expression = expression + "(#{index_expression_identifier} == #{index} ? #{receiver}.field_#{index} : "
                end

                # Out of bounds case should throw and exception
                expression = expression + "NULL"

                for index in 0...@fields.size
                    expression = expression + ")"
                end

                return expression
            end
        end
    end
end
