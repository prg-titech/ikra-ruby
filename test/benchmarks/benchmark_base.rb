require "test/unit"
require "ikra"

module BenchmarkBase
    COLLECT_DATA = false
    RUNS = 5
    DATA_OUTPUT_DIR = "/home/matthias/ikra-ruby/test/benchmarks/ppl2017/data/"

    def setup_in_test
        Ikra::Configuration.reset_state
        Ikra::Translator::CommandTranslator::ProgramBuilder::Launcher.reset_time

        test_name = "benchmark_" + self.class.to_s
        file_name = Ikra::Configuration.log_file_name_for(test_name)
        File.delete(file_name) if File.exist?(file_name)
        Ikra::Log.reopen(file_name)
    end

    def execute
        raise NotImplementedError.new("abstract method")
    end

    def expected
        return nil
    end

    def test_run
        if COLLECT_DATA
            runs = RUNS
        else
            runs = 1
        end

        fastest_kernel = -1
        fastest_detail = []

        for run in 0...runs
            setup_in_test
            Ikra::Configuration.codegen_expect_file_name = "benchmark_" + self.class.to_s

            header = """
--------------------------------------------------------------------------------
Benchmark (\##{run}):                    #{self.class.to_s}"""

            Ikra::Log.info(header)
            puts header

            start_entire = Time.now
            result = execute
            result.execute
            end_entire = Time.now

            stats_data = print_stats

            # Add rest interpeter (no FFI, no compilation, no external)
            stats_data.push((end_entire - start_entire) - stats_data[-1] - stats_data[-2] - stats_data.first)

            stats = """--------------------------------------------------------------------------------
Rest Ruby Interpreter:             #{'%.04f' % stats_data.last}
--------------------------------------------------------------------------------
Total Execution Time:              #{'%.04f' % (end_entire - start_entire)}
--------------------------------------------------------------------------------
"""
            stats_data.push(end_entire - start_entire)

            Ikra::Log.info(stats)
            puts stats

            if fastest_kernel == -1 || stats_data[1] < fastest_kernel
                # Update fastest collected data
                fastest_kernel = stats_data[1]
                fastest_detail = stats_data
            end

            if expected != nil
                # Check if result is correct
                assert_equal(expected, result.to_a)

                Ikra::Log.info("Equality check passed")
                puts "Equality check passed"
            end

            puts "\n\n"
        end

        if COLLECT_DATA
            puts "Fastest collect data: Kernel time #{fastest_kernel}"

            selected_stats_data = [fastest_detail[1], fastest_detail[5], fastest_detail[2], fastest_detail[3], fastest_detail[4], fastest_detail[8], fastest_detail[0]]
            puts "Details: #{selected_stats_data}"

            File.write(
                DATA_OUTPUT_DIR + self.class.to_s,
                selected_stats_data.join("\n"))
        end
    end

    def print_stats
        launcher = Ikra::Translator::CommandTranslator::ProgramBuilder::Launcher

        # Remaining time im host section
        rest_host = launcher.last_time_total_external - launcher.last_time_setup_cuda - launcher.last_time_kernel - launcher.last_time_allocate_memory - launcher.last_time_transfer_memory - launcher.last_time_free_memory

        stats = """--------------------------------------------------------------------------------
NVCC Time:                         #{'%.04f' % launcher.last_time_compiler}
--------------------------------------------------------------------------------
CUDA Setup Time:                   #{'%.04f' % launcher.last_time_setup_cuda}
Kernel(s) Time:                    #{'%.04f' % launcher.last_time_kernel}
Allocate Memory Time:              #{'%.04f' % launcher.last_time_allocate_memory}
Transfer Memory Time:              #{'%.04f' % launcher.last_time_transfer_memory}
Free Memory Time:                  #{'%.04f' % launcher.last_time_free_memory}
Rest Host Section:                 #{'%.04f' % rest_host}

Total External Execution Time:     #{'%.04f' % launcher.last_time_total_external}
--------------------------------------------------------------------------------
Read Result (FFI) Time:            #{'%.04f' % launcher.last_time_read_result_ffi}
"""

        Ikra::Log.info(stats)
        puts stats

        return [launcher.last_time_compiler, launcher.last_time_kernel, launcher.last_time_allocate_memory, launcher.last_time_transfer_memory, launcher.last_time_free_memory, rest_host, launcher.last_time_total_external, launcher.last_time_read_result_ffi]
    end
end
