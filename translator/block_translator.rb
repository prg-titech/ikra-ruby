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

        # The result of Ruby-to-CUDA translation of a block using {Translator}
        class BlockTranslationResult

            # @return [String] Generated CUDA source code
            attr_accessor :block_source

            # @return [UnionType] Return value type of method/block
            attr_accessor :result_type

            # @return [String] Name of function in CUDA source code
            attr_accessor :function_name

            # @return [Array<Ikra::AST::MethodDefinition>] Auxiliary methods that are called by this block (including transitive method calls)
            attr_accessor :aux_methods

            def initialize(c_source:, result_type:, function_name:, aux_methods: [])
                @block_source = c_source
                @result_type = result_type
                @function_name = function_name
                @aux_methods = aux_methods
            end

            def generated_source
                @aux_methods.map do |meth|
                    meth.to_c_source
                end.join("\n\n") + @block_source
            end
        end

        BlockSelectorDummy = :"<BLOCK>"

        class << self
            # Translates a Ruby block to CUDA source code.
            # @param [AST::Node] ast abstract syntax tree of the block
            # @param [EnvironmentBuilder] env_builder environment builder instance collecting information about lexical variables (environment)
            # @param [Hash{Symbol => UnionType}] input_types types of arguments passed to the block
            # @param [Hash{Symbol => Object}] lexical_variables all lexical variables that are accessed within the block
            # @return [BlockTranslationResult]
            def translate_block(ast:, env_builder:, block_parameter_types: {}, lexical_variables: {})
                Log.info("Translating block with input types #{input_types.to_type_array_string}")

                increase_translation_id

                # Define MethodDefinition for block
                block_def = AST::MethodDefinition.new(
                    type: Types::UnionType.new,        # TODO: what to pass in here?
                    selector: BlockSelectorDummy,
                    parameter_variables: block_parameter_types,
                    return_type: Types::UnionType.new,
                    ast: ast)

                # Lexical variables
                lexical_variables.each do |name, value|
                    block_def.lexical_variables[name] = Types::UnionType.new(value.class.to_ikra_type)
                end

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                return_type = type_inference_visitor.process_method(block_def)
                aux_methods = type_inference_visitor.methods
                    
                # Translate to CUDA/C++ code
                translation_result = ast.translate_statement

                # Load environment variables
                lexical_variables.each do |name, value|
                    mangled_name = mangle_var_name_translation_id(name)
                    translation_result.prepend("#{value.class.to_ikra_type.to_c_type} #{name} = #{EnvParameterName}->#{mangled_name};\n")

                    env_builder.add_variable(
                        var_name: mangled_name, 
                        type: type, 
                        value: lexical_variables[name])
                end

                # Declare local variables
                block_def.local_variables.each do |name, types|
                    translation_result.prepend("#{types.singleton_type.to_c_type} #{name};\n")
                end

                # Function signature
                mangled_name = mangle_block_name_translation_id("")

                if not return_type.is_singleton?
                    raise "Cannot handle polymorphic return types yet"
                end

                function_parameters = ["environment_t *#{EnvParameterName}"]
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