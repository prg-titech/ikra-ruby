module Ikra
    module Translator
        class CommandTranslator
            class WhileLoopKernelLauncher < KernelLauncher
                def initialize(
                    kernel_builder:,
                    condition:,
                    before_loop: "",
                    post_iteration: "")

                    super(kernel_builder)
                    @condition = condition
                    @before_loop = before_loop
                    @post_iteration = post_iteration
                end

                attr_reader :condition
                attr_reader :before_loop
                attr_reader :post_iteration

                def build_kernel_launcher
                    Log.info("Building for-loop kernel launcher")

                    assert_ready_to_build

                    result = ""
                    result = result + before_loop + "\n"
                    result = result + "while (#{condition}) {\n"
                    result = result + super
                    result = result + "\n" + post_iteration
                    result = result + "\n}\n"

                    return result
                end
            end
        end
    end
end
