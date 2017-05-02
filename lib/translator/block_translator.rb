require_relative "../ast/nodes.rb"
require_relative "../ast/builder.rb"
require_relative "../types/types"
require_relative "../parsing"
require_relative "../ast/printer"
require_relative "variable_classifier_visitor"

module Ikra
    module Translator

        # The result of Ruby-to-CUDA translation of a block using {Translator}
        class BlockTranslationResult

            # @return [String] Generated CUDA source code
            attr_accessor :block_source

            # @return [UnionType] Return value type of method/block
            attr_accessor :result_type

            # @return [String] Name of function of block in CUDA source code
            attr_accessor :function_name

            # @return [String] Auxiliary methods that are called by this block 
            # (including transitive method calls)
            attr_accessor :aux_methods

            def initialize(c_source:, result_type:, function_name:, aux_methods: [])
                @block_source = c_source
                @result_type = result_type
                @function_name = function_name
                @aux_methods = aux_methods
            end
        end

        BlockSelectorDummy = :"<BLOCK>"

        class << self
            # Translates a Ruby block to CUDA source code.
            # @param [AST::BlockDefNode] block_def_node AST (abstract syntax tree) of the block
            # @param [EnvironmentBuilder] environment_builder environment builder instance 
            # collecting information about lexical variables (environment)
            # @param [Array{Variable}] block_parameters types and names of parameters 
            # to the block
            # @param [Hash{Symbol => Object}] lexical_variables all lexical variables that are 
            # accessed within the block
            # @param [Integer] command_id a unique identifier of the block
            # @param [String] pre_execution source code that should be run before executing the
            # block
            # @param [Array{Variable}] override_block_parameters overrides the the declaration of
            # parameters that this block accepts.
            # @param [EntireInputTranslationResult] entire_input_translation The result of 
            # `translate_entire_input`
            # @return [BlockTranslationResult]
            def translate_block(
                block_def_node:, 
                environment_builder:, 
                command_id:, 
                lexical_variables: {}, 

                # One one of the two following parameter configurations is valid:
                # a) Either this parameter is given:
                entire_input_translation: nil,

                # b) or these parameters are given (some are optional):
                pre_execution: nil, 
                override_block_parameters: nil,
                block_parameters: nil)

                # Check and prepare arguments
                if pre_execution != nil and entire_input_translation != nil
                    raise ArgumentError.new("pre_execution and entire_input_translation given")
                elsif entire_input_translation != nil
                    pre_execution = entire_input_translation.pre_execution
                elsif pre_execution == nil
                    pre_execution = ""
                end

                if block_parameters != nil and entire_input_translation != nil
                    raise ArgumentError.new("block_parameters and entire_input_translation given")
                elsif entire_input_translation != nil
                    block_parameters = entire_input_translation.block_parameters
                elsif block_parameters == nil
                    block_parameters = []
                end

                if override_block_parameters != nil and entire_input_translation != nil
                    raise ArgumentError.new("override_block_parameters and entire_input_translation given")
                elsif entire_input_translation != nil
                    override_block_parameters = entire_input_translation.override_block_parameters
                elsif override_block_parameters == nil
                    override_block_parameters = block_parameters
                end


                # Build hash of parameter name -> type mappings
                block_parameter_types = {}
                for variable in block_parameters
                    block_parameter_types[variable.name] = variable.type
                end

                parameter_types_string = "[" + block_parameter_types.map do |id, type| "#{id}: #{type}" end.join(", ") + "]"
                Log.info("Translating block with input types #{parameter_types_string}")

                # Add information to block_def_node
                block_def_node.parameters_names_and_types = block_parameter_types

                # Lexical variables
                lexical_variables.each do |name, value|
                    block_def_node.lexical_variables_names_and_types[name] = value.ikra_type.to_union_type
                end

                # Type inference
                type_inference_visitor = TypeInference::Visitor.new
                return_type = type_inference_visitor.process_block(block_def_node)
                
                # Translation to source code
                ast_translator = ASTTranslator.new

                # Auxiliary methods are instance methods that are called by the block
                aux_methods = type_inference_visitor.all_methods.map do |method|
                    ast_translator.translate_method(method)
                end

                # Generate method predeclarations
                aux_methods_predecl = type_inference_visitor.all_methods.map do |method|
                    ast_translator.translate_method_predecl(method)
                end

                # Start with predeclarations
                aux_methods = aux_methods_predecl + aux_methods

                # Classify variables (lexical or local)
                block_def_node.accept(VariableClassifier.new(
                    lexical_variable_names: lexical_variables.keys))

                # Translate to CUDA/C++ code
                translation_result = ast_translator.translate_block(block_def_node)

                # Load environment variables
                lexical_variables.each do |name, value|
                    type = value.ikra_type
                    mangled_name = environment_builder.add_object(name, value)
                    translation_result.prepend("#{type.to_c_type} #{Constants::LEXICAL_VAR_PREFIX}#{name} = #{Constants::ENV_IDENTIFIER}->#{mangled_name};\n")
                end

                # Declare local variables
                block_def_node.local_variables_names_and_types.each do |name, type|
                    translation_result.prepend("#{type.to_c_type} #{name};\n")
                end

                # Function signature
                mangled_name = "_block_k_#{command_id}_"

                function_parameters = ["environment_t *#{Constants::ENV_IDENTIFIER}"]

                parameter_decls = override_block_parameters.map do |variable|
                    "#{variable.type.to_c_type} #{variable.name}"
                end

                function_parameters.push(*parameter_decls)

                translation_result = Translator.read_file(
                    file_name: "block_function_head.cpp",
                    replacements: { 
                        "name" => mangled_name, 
                        "result_type" => return_type.to_c_type,
                        "parameters" => function_parameters.join(", "),
                        "body" => wrap_in_c_block(pre_execution + "\n" + translation_result)})

                # TODO: handle more than one result type
                return BlockTranslationResult.new(
                    c_source: translation_result, 
                    result_type: return_type,
                    function_name: mangled_name,
                    aux_methods: aux_methods)
            end
        end
    end
end
