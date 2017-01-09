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
            result = input

            for i in 0...100
                if i % 2 == 0
                    # Branch 1
                    result = result.pmap(1337) do |j|
                        j + 1
                    end

                    result2 = proc do |j|
                        j+2
                    end
                else
                    # Branch 2
                    result = result.pmap do |j|
                        j + 10
                    end
                end
            end

            result
        end

        section_result[0]
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
    end
end
