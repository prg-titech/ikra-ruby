require "logger"
require "test/unit"

class UnitTestCase < Test::Unit::TestCase
    def setup
        Ikra::Configuration.codegen_expect_file_name = nil

        test_name = self.class.to_s + "\#" + method_name
        file_name = Ikra::Configuration.log_file_name_for(test_name)
        File.delete(file_name) if File.exist?(file_name)
        Ikra::Log.reopen(file_name)
    end
end