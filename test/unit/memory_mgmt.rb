require "ikra"
require_relative "unit_test_template"

class MemoryManagementTest < UnitTestCase
    def test_manual_memory_free
        array_gpu = Array.pnew(51100) do |j|
            j + 1
        end

        array_cpu = Array.new(51100) do |j|
            j + 1000
        end

        section_result = Ikra::Symbolic.host_section(array_gpu) do |input|

            a = input
            old_val = a

            for i in 1...1000
                old_val = a
                a = a.pmap do |k|
                    k + 1
                end

                old_val.free_memory
            end

            a
        end

        assert_equal(array_cpu.reduce(:+) , section_result.reduce(:+))
    end
end
