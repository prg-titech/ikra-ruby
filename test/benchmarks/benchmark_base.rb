require "test/unit"
require "ikra"

module BenchmarkBase
    def setup
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
        Ikra::Configuration.codegen_expect_file_name = "benchmark_" + self.class.to_s

        header = """
--------------------------------------------------------------------------------
Benchmark:                         #{self.class.to_s}"""

        Ikra::Log.info(header)
        puts header

        start_entire = Time.now
        result = execute
        result.execute
        end_entire = Time.now

        print_stats
        stats = """Total Execution Time:              #{'%.04f' % (end_entire - start_entire)}
--------------------------------------------------------------------------------
"""
        Ikra::Log.info(stats)
        puts stats

        if expected != nil
            # Check if result is correct
            assert_equal(expected, result.to_a)

            Ikra::Log.info("Equality check passed")
            puts "Equality check passed"
        end

        puts "\n\n"
    end

    def print_stats
        launcher = Ikra::Translator::CommandTranslator::ProgramBuilder::Launcher

        stats = """--------------------------------------------------------------------------------
NVCC Time:                         #{'%.04f' % launcher.last_time_compiler}
--------------------------------------------------------------------------------
CUDA Setup Time:                   #{'%.04f' % launcher.last_time_setup_cuda}
Kernel(s) Time:                    #{'%.04f' % launcher.last_time_kernel}
Allocate Memory Time:              #{'%.04f' % launcher.last_time_allocate_memory}
Transfer Memory Time:              #{'%.04f' % launcher.last_time_transfer_memory}
Free Memory Time:                  #{'%.04f' % launcher.last_time_free_memory}

Total External Execution Time:     #{'%.04f' % launcher.last_time_total_external}
--------------------------------------------------------------------------------
Read Result (FFI) Time:            #{'%.04f' % launcher.last_time_read_result_ffi}
--------------------------------------------------------------------------------
"""

        Ikra::Log.info(stats)
        puts stats
    end
end
