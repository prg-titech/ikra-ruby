require "rbconfig"

module Ikra
    module Configuration
        class << self
            SUPPORTED_OS = [:linux, :macosx]
            CUDA_NVCC = "/usr/local/cuda-8.0/bin/nvcc"
            CUDA_COMMON_INCLUDE = "/home/matthias/NVIDIA_CUDA-7.5_Samples/common/inc"
            CUDA_CUPTI_INCLUDE = "/usr/local/cuda-7.5/extras/CUPTI/include"

            def check_software_configuration
                if !SUPPORTED_OS.include?(operating_system)
                    raise "Operating system not supported: #{operating_system}"
                end

                # Check if nvcc is installed
                %x(#{CUDA_NVCC})
                if $?.exitstatus != 1
                    raise "nvcc not installed"
                end

                if !File.directory?(CUDA_COMMON_INCLUDE)
                    raise "Directory does not exist: #{CUDA_COMMON_INCLUDE}. Check OS configuration!"
                end

                if !File.directory?(CUDA_CUPTI_INCLUDE)
                    raise "Directory does not exist: #{CUDA_CUPTI_INCLUDE}. Check OS configuration!"
                end
            end

            def nvcc_invocation_string(in_file, out_file)
                "#{CUDA_NVCC} -o #{out_file} -I#{CUDA_COMMON_INCLUDE} -I#{CUDA_CUPTI_INCLUDE} --shared -Xcompiler -fPIC #{in_file}"
            end

            def so_suffix
                if operating_system == :linux
                    "so"
                elsif operating_system == :macosx
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
