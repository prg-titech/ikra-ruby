require_relative "../ast/nodes.rb"
require_relative "../ast/builder.rb"
require_relative "../ast/translator.rb"
require_relative "../types/type_inference"
require_relative "../types/primitive_type"
require_relative "../parsing"
require_relative "../scope"
require_relative "translator"
require_relative "last_returns_visitor"
require_relative "local_variables_enumerator"
require_relative "../ast/printer"
require_relative "../ast/method_definition"

module Ikra
    module Translator
        class BlockTranslationResult
            attr_accessor :c_source
            attr_accessor :result_type
            attr_accessor :function_name
            attr_accessor :aux_methods

            def initialize(c_source:, result_type:, function_name:, aux_methods: [])
                @c_source = c_source
                @result_type = result_type
                @function_name = function_name
                @aux_methods = aux_methods
            end
        end

        block_selector_dummy = :"<BLOCK>"

        class << self
            def translate_block(block:, symbol_table:, env_builder:, input_types: [])
                Log.info("Translating block with input types #{input_types.to_type_array_string}")

                increase_translation_id

                # Generate AST
                parser_local_vars = block.binding.local_variables + block_parameters
                source = Parsing.parse_block(block, parser_local_vars)
                ast = AST::Builder.from_parser_ast(source)

                # Block parameters and types
                block_parameters = block.parameters.map do |param|
                    param[1]
                end
                block_parameter_types = Hash[*block_parameters.zip(input_types).flatten]

                # Define MethodDefinition for block
                block_def = AST::MethodDefinition.new(
                    type: UnionType.new,        # TODO: what to pass in here?
                    selector: block_selector_dummy,
                    parameter_variables: block_parameter_types,
                    return_type: UnionType.new,
                    ast: ast)

                # Lexical variables
                block.binding.local_variables.each do |var|
                    block_def.lexical_variables[var] = UnionType.new(block.binding.local_variable_get(var).class.to_ikra_type)
                end

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                type_inference_visitor.process_method(block_def)
                aux_methods = type_inference_visitor.methods
                return_type = block_def.symbol_table.top_frame.return_type



                # ------------ TODO: continue refactoring here ----------------
                # Get required env variables from second to top-most frame
                env_variables = (block_def.symbol_table.read_and_written_variables(-1) - block_parameters).map do |var|
                    VariableWithType.new(var_name: var, type: block_def.symbol_table.get_type(var))
                end
                    
                # Translate to CUDA/C++ code
                translation_result = ast.translate_statement

                # Load environment variables
                env_variables.each do |var|
                    mangled_name = mangle_var_name_translation_id(var.var_name)
                    if not var.type.is_singleton?
                        raise "Cannot handle polymorphic types yet"
                    end
                    translation_result.prepend("#{var.type.singleton_type.to_c_type} #{var.var_name} = #{EnvParameterName}->#{mangled_name};\n")

                    env_builder.add_variable(var_name: mangled_name, 
                        type: var.type, 
                        value: block.binding.local_variable_get(var.var_name))
                end

                # Declare local variables
                local_variables.each do |name, types|
                    translation_result.prepend("#{types.singleton_type.to_c_type} #{name};\n")
                end

                # Function signature
                mangled_name = mangle_block_name_translation_id("")

                if not return_type.is_singleton?
                    raise "Cannot handle polymorphic return types yet"
                end

                function_parameters = ["struct #{EnvStructName} *#{EnvParameterName}"]
                block_parameter_types.each do |param|
                    function_parameters.push("#{param[1].singleton_type.to_c_type} #{param[0].to_s}")
                end

                translation_result = "__device__ #{return_type.singleton_type.to_c_type} #{mangled_name}(#{function_parameters.join(", ")})\n" +
                    wrap_in_c_block(translation_result)

                # TODO: handle more than one result type
                BlockTranslationResult.new(
                    c_source: translation_result, 
                    result_type: return_type,
                    function_name: mangled_name,
                    aux_methods: aux_methods)
            end
        end
    end
end