require "set"
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

module Ikra
    module Translator
        class BlockTranslationResult
            attr_accessor :c_source
            attr_accessor :result_types
            attr_accessor :function_name
            attr_accessor :aux_methods

            def initialize(c_source:, result_types:, function_name:, aux_methods: [])
                @c_source = c_source
                @result_types = result_types
                @function_name = function_name
                @aux_methods = aux_methods
            end
        end

        class << self
            def translate_block(block:, symbol_table:, env_builder:, input_types: [])
                Log.info("Translating block with input types #{input_types.to_type_array_string}")

                translation_result = nil
                env_variables = nil
                return_types = nil
                local_variables = nil
                aux_methods = nil

                increase_translation_id

                # Block parameters and types
                block_parameters = block.parameters.map do |param|
                    param[1]
                end
                block_parameter_types = block_parameters.zip(input_types)

                symbol_table.new_frame do
                    # Generate AST
                    parser_local_vars = block.binding.local_variables + block_parameters
                    source = Parsing.parse_block(block, parser_local_vars)
                    ast = Ikra::AST::Builder.from_parser_ast(source)

                    # Add return statements
                    ast.accept(Ikra::Translator::LastStatementReturnsVisitor.new)

                    # Add lexical variables to symbol table
                    block.binding.local_variables.each do |var|
                        symbol_table.add_types(var, [block.binding.local_variable_get(var).class.to_ikra_type].to_set)
                    end

                    # Add block parameter to symbol table
                    # TODO: find a good way to pass type in
                    block_parameter_types.each do |var|
                        symbol_table.add_types(var[0], var[1])
                    end

                    # Infer type of all statements
                    symbol_table.new_frame do
                        symbol_table.top_frame.function_frame!
                        type_inference_visitor = Ikra::TypeInference::Visitor.new(symbol_table, block.binding)
                        ast.accept(type_inference_visitor)

                        aux_methods = type_inference_visitor.aux_methods
                        return_types = symbol_table.top_frame.return_types
                    end

                    # Get required env variables
                    env_variables = (symbol_table.read_and_written_variables(-1) - block_parameters).map do |var|
                        VariableWithType.new(var_name: var, type: symbol_table.get_types(var))
                    end

                    # Get local variables
                    local_variables_enumerator = LocalVariablesEnumerator.new
                    ast.accept(local_variables_enumerator)
                    local_variables = local_variables_enumerator.local_variables.reject do |name, type|
                        block_parameters.include?(name) ||  # No input parameters
                            symbol_table.read_and_written_variables(-1).include?(name) # No env vars
                    end
                    
                    # Translate to CUDA/C++ code
                    translation_result = ast.translate_statement
                end

                # Load environment variables
                env_variables.each do |var|
                    mangled_name = mangle_var_name_translation_id(var.var_name)
                    if var.type.size != 1
                        raise "Cannot handle != 1 env argument types"
                    end
                    translation_result.prepend("#{var.type.first.to_c_type} #{var.var_name} = #{EnvParameterName}->#{mangled_name};\n")

                    env_builder.add_variable(var_name: mangled_name, 
                        types: var.type, 
                        value: block.binding.local_variable_get(var.var_name))
                end

                # Declare local variables
                local_variables.each do |name, types|
                    translation_result.prepend("#{types.first.to_c_type} #{name};\n")
                end

                # Function signature
                mangled_name = mangle_block_name_translation_id("")

                if return_types.size != 1
                    raise "Cannot handle #{return_types.size} return types"
                end
                return_type = return_types.first

                function_parameters = ["struct #{EnvStructName} *#{EnvParameterName}"]
                block_parameter_types.each do |param|
                    function_parameters.push("#{param[1].first.to_c_type} #{param[0].to_s}")
                end

                translation_result = "__device__ #{return_type.to_c_type} #{mangled_name}(#{function_parameters.join(", ")})\n" +
                    wrap_in_c_block(translation_result)

                # TODO: handle more than one result type
                BlockTranslationResult.new(c_source: translation_result, 
                    result_types: [return_types],
                    function_name: mangled_name,
                    aux_methods: aux_methods)
            end
        end
    end
end