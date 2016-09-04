require_relative "../ast/nodes.rb"
require_relative "../ast/builder.rb"
require_relative "../ast/translator.rb"
require_relative "../types/type_inference"
require_relative "../types/primitive_type"
require_relative "../parsing"
require_relative "../scope"
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
            # @param [EnvironmentBuilder] environment_builder environment builder instance collecting information about lexical variables (environment)
            # @param [Hash{Symbol => UnionType}] block_parameter_types types of arguments passed to the block
            # @param [Hash{Symbol => Object}] lexical_variables all lexical variables that are accessed within the block
            # @param [Fixnum] command_id a unique identifier of the block
            # @return [BlockTranslationResult]
            def translate_block(ast:, environment_builder:, command_id:, block_parameter_types: {}, lexical_variables: {})
                parameter_types_string = "[" + block_parameter_types.map do |id, type| "#{id}: #{type}" end.join(", ") + "]"
                Log.info("Translating block with input types #{parameter_types_string}")

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
                # The following method returns nested dictionaries, but we only need the values
                aux_methods = type_inference_visitor.methods.values.map do |hash|
                    hash.values
                end.flatten

                # Translate to CUDA/C++ code
                translation_result = ast.translate_statement

                # Load environment variables
                lexical_variables.each do |name, value|
                    type = value.class.to_ikra_type
                    mangled_name = environment_builder.add_object(name, value)
                    translation_result.prepend("#{type.to_c_type} #{name} = #{Constants::ENV_IDENTIFIER}->#{mangled_name};\n")
                end

                # Declare local variables
                block_def.local_variables.each do |name, types|
                    translation_result.prepend("#{types.singleton_type.to_c_type} #{name};\n")
                end

                # Function signature
                mangled_name = "_block_k_#{command_id}_"

                if not return_type.is_singleton?
                    raise "Cannot handle polymorphic return types yet"
                end

                function_parameters = ["environment_t *#{Constants::ENV_IDENTIFIER}"]
                block_parameter_types.each do |param|
                    function_parameters.push("#{param[1].to_c_type} #{param[0].to_s}")
                end

                function_head = Translator.read_file(
                    file_name: "block_function_head.cpp",
                    replacements: { 
                        "name" => mangled_name, 
                        "return_type" => return_type.singleton_type.to_c_type,
                        "parameters" => function_parameters.join(", ")})

                translation_result = function_head + wrap_in_c_block(translation_result)

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