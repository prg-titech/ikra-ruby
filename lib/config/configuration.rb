require_relative "../symbolic/symbolic"

module Ikra
    module Configuration
        JOB_REORDERING = true
        @@expect_file_name = "last_generated"

        def self.resource_file_name(file_name)
            File.expand_path("resources/cuda/#{file_name}", File.dirname(File.dirname(File.expand_path(__FILE__))))
        end

        def self.codegen_expect_file_name_for(file_name)
            File.expand_path("gen/codegen_expect/#{file_name}", File.dirname(File.dirname(File.dirname(File.expand_path(__FILE__)))))
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