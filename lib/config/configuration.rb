module Ikra
    module Configuration
        JOB_REORDERING = true
        @@expect_file_name = "last_generated"

        def self.resource_file_name(file_name)
        	File.expand_path("resources/cuda/#{file_name}", File.dirname(File.dirname(File.expand_path(__FILE__))))
        end

        def self.codegen_expect_file_name
        	File.expand_path("gen/codegen_expect/#{@@expect_file_name}.cu", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__)))))
        end

        def self.codegen_expect_file_name=(value)
        	@@expect_file_name = value
        end
    end
end