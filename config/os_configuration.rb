require "rbconfig"

module Ikra
    module Configuration
        class << self
            def check_software_configuration
                if operating_system != :linux
                    raise "Operating system not supported"
                end

                # Check if nvcc is installed
                %x("nvcc")
                if $?.exitstatus != 1
                    raise "nvcc not installed"
                end
            end

            def nvcc_invocation_string(in_file, out_file)
                "nvcc -o #{out_file} -I/usr/local/cuda/samples/common/inc --shared -Xcompiler -fPIC #{in_file}"
            end

            def so_suffix
                if operating_system == :linux
                    "so"
                elsif operating_system == :windows
                    "dll"
                else
                    raise "Operating system not supported"
                end
            end

            def operating_system
                # copied from: http://stackoverflow.com/questions/11784109/detecting-operating-systems-in-ruby

                host_os = RbConfig::CONFIG['host_os']
                case host_os
                    when /mswin|msys|mingw|cygwin|bccwin|wince|emc/
                        :windows
                    when /darwin|mac os/
                        :macosx
                    when /linux/
                        :linux
                    when /solaris|bsd/
                        :unix
                    else
                        raise "Unknown operating system"
                end
            end
        end
    end
end