require "ikra"

def test_host_section_iterative_map
    array_gpu = Array.pnew(511) do |j|
        j + 1
    end

    array_cpu = Array.new(511) do |j|
        j + 10
    end

    section_result = Ikra::Symbolic.host_section(array_gpu) do |input|

        a = input

        for i in 1...10
            a = a.pmap do |k|
                k + 1
            end
        end

        a
    end

    if array_cpu.reduce(:+) == section_result.reduce(:+)
        Ikra::Log.info("Test passed!")
    else
        # Use unit test for debugging
        raise "Test failed (check unit test in host_section_test.rb)"
    end
end