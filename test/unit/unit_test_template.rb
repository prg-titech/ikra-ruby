require "test/unit"

class UnitTestCase < Test::Unit::TestCase
    def setup
        Ikra::Configuration.codegen_expect_file_name = nil
    end
end