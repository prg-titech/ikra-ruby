module Ikra
    module Translator
        class CommandTranslator
            class ForLoopKernelLauncher < KernelLauncher
                def initialize(
                    kernel_builder:, 
                    from_expr: "0", 
                    to_expr:, 
                    var_name: "i", 
                    before_loop: "")

                    super(kernel_builder)
                    @from_expr = from_expr
                    @to_expr = to_expr
                    @var_name = var_name
                    @before_loop = before_loop
                end

                attr_reader :from_expr
                attr_reader :to_expr
                attr_reader :var_name
                attr_reader :before_loop

                def build_kernel_launcher
                    Log.info("Building for-loop kernel launcher")

                    assert_ready_to_build

                    result = before_loop + "\n"
                    result = result + "for (int #{var_name} = #{from_expr}; #{var_name}  < #{to_expr}; #{var_name} ++)\n{"

                    result = result + super
                    result = result + "\n}\n"

                    return result
                end
            end
        end
    end
end
