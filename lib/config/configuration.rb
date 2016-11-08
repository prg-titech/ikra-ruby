require 'fileutils'

require_relative "../symbolic/symbolic"

module Ikra
    module Configuration
        JOB_REORDERING = true

        @@expect_file_name = "last_generated"

        class << self
            # For debug purposes only: provide different CUDA source code file for compilation
            attr_accessor :override_cuda_file
            attr_accessor :kernel_iterations
        end

        def self.resource_file_name(file_name)
            File.expand_path("resources/cuda/#{file_name}", File.dirname(File.dirname(File.expand_path(__FILE__))))
        end

        def self.codegen_expect_file_name_for(file_name)
            FileUtils.mkdir_p(File.expand_path("gen/codegen_expect", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__))))))
            
            File.expand_path("gen/codegen_expect/#{file_name}", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__)))))
        end

        def self.log_file_name_for(test_case_name)
            FileUtils.mkdir_p(File.expand_path("gen/log", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__))))))

            File.expand_path("gen/log/#{test_case_name}.log", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__)))))
        end

        def self.codegen_expect_file_name
            if @@expect_file_name == nil
                # Do not generate expect file
                return nil
            end

            return codegen_expect_file_name_for(@@expect_file_name + ".cu")
        end

        def self.codegen_expect_file_name=(value)
            @@expect_file_name = value
        end

        def self.reset_state
            Symbolic::ArrayCommand.reset_unique_id
        end
    end
end