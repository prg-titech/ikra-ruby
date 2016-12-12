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
            def generate_read(receiver, selector, *args)
                # Type inference already ensured that there is exactly one parameter which is
                # an IntLiteral.

                return "#{receiver}.field_#{args.first.value}"
            end
        end
    end
end
