require "ikra"
require_relative "unit_test_template"

class HostSectionTest < UnitTestCase
    def test_simple_host_section
        array_gpu = Array.pnew(511) do |j|
            j + 1
        end

        array_cpu = Array.new(511) do |j|
            j + 2
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
            input.pmap do |k|
                k + 1
            end
        end

        section_result[0]


        # section_result = Ikra::Symbolic.host_section(array_gpu) do |input|
        #     result = input
        # 
        #     for i in 0...100
        #         if i % 2 == 0
        #             # Branch 1
        #             result = result.pmap do |j|
        #                 j + 1
        #             end
        #
        #         else
        #             # Branch 2
        #             result = result.pmap do |j|
        #                 j + 10
        #             end
        #         end
        #     end
        #
        #     result
        # end

        # section_result[0]
        # --> Returns ArrayHostSectionCommand, `execute` sets a result pointer field if keep is
        # set to `true`.

        # CUDA section --> host function

        # Should be translated to the following CUDA program:
        #
        # int * section_1(int * input) {
        #     int * result = input;
        #     for (int i = 0; i < 100; i++) {
        #         if (i % 2 == 0) {
        #             kernel_branch_1(result, result);
        #         } else {
        #             kernel_branch_2(result, result);
        #         }
        #     }
        #     return result;
        # }

        # If result is combined with another command:
        # next_result = section_result.pmap do |i| i + 1 end
        # Somehow integrate with kernel_builder etc

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))



        array_gpu = Array.pnew(512) do |j|
            j + 1 + 0
        end

        array_gpu_2 = Array.pnew(1024) do |j|
            j + 1
        end

        # Another problematic case:
        #
        # section_result = Ikra::Symbolic.host_section (array_gpu, array_gpu_2) do |i1, i2|
        #     r = i1
        #     if ...
        #         r = i2
        #     end
        # 
        #     r.pmap do |i|
        #         i + 4
        #     end
        # end
        #
        # What is the type of r?
        # - The type of r has to be reified and contain the size of `r` and a pointer to `r` at 
        #   the very least
        # - Or maybe this can all the handled by the normal type inference rules that we have now?
        # - Cannot symbolically execute such statements (or can it?)
        # - Symbolically execute parts inside loops (separate CFG)
    end
end
