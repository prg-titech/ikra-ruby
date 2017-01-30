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

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))

    end
end
