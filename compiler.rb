require_relative "translator"

module Compiler
    class CompilationRequest
        attr_accessor :block
        attr_accessor :size
        
        # Array of InputVariable
        attr_reader :input_vars
        
        def initialize(block: block, size: size)
            @block = block
            @size = size
            @input_vars = []
        end

        def add_input_var(var)
            @input_vars.push(var)
        end
    end
    
    class KernelCompilationResult
        attr_accessor :c_source
        
        EnvironmentVariable = "_env_"
        
        def initialize
            @block_index = 0
            @previous_block_result_types = []
            @invocation_source = ""
            @kernel_inner_source = ""
            @env_offset = 0
        end
        
        def merge_request!(request)
            # Fill in types from previous translation
            num_fusion_request = request.input_vars.select do |var|
                var.is_fusion?
            end.size
            if num_fusion_request != @previous_block_result_types.size
                raise "Mismatch in number of resulting values (expected #{num_fusion_request}, found #{@previous_block_result_types.size}"
            end
            
            previous_type_index = 0
            request.input_vars.each do |var|
                if var.is_fusion?
                    var.type = @previous_block_result_types[previous_type_index]
                    previous_type_index += 1
                end
            end
            
            # Translate next block
            block_result = Translator.translate_block(
                block: request.block, 
                size: request.size, 
                input_vars: request.input_vars, 
                function_name: "kernel_inner_#{@block_index}")
            
            @kernel_inner_source = block_result.c_source
            @invocation_source = "kernel_inner_#{@block_index}(#{@invocation_source}, (void *) (((char *) #{EnvironmentVariable}) + #{@env_offset}))"
            
            @previous_block_result_types = block_result.result_type
            @env_offset += block_result.env_size
            @block_index += 1
        end
        
        def full_source
            @kernel_inner_source + "\n\n" + """__global__ kernel()
{
    return #{@invocation_source};
}
"""
        end

        def wrap_in_c_block(str)
            "{\n" + str.split("\n").map do |line| "    " + line end.join("\n") + "\n}\n"
        end
    end
    
    def self.compile(request)
        compilation_result = KernelCompilationResult.new
        compilation_result.merge_request!(request)
        compilation_result
    end
end