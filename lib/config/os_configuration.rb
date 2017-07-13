require "rbconfig"
require "logger"

module Ikra
    module Configuration

        # These values are used only if auto configuration fails
        @cuda_manual_nvcc = "/usr/local/cuda-8.0/bin/nvcc"
        @cuda_manual_common_include = "/home/matthias/NVIDIA_CUDA-7.5_Samples/common/inc"
        @cuda_manual_cupti_include = "/usr/local/cuda-7.5/extras/CUPTI/include"

        @is_initialized = false
        @auto_config = true

        class << self
            SUPPORTED_OS = [:linux, :macosx]
            ATTEMPT_AUTO_CONFIG = true

            attr_accessor :cuda_manual_nvcc
            attr_accessor :cuda_manual_common_include
            attr_accessor :cuda_manual_cupti_include
            attr_accessor :auto_config

            def is_initialized?
                return @is_initialized
            end

            def reinitialize!
                if !SUPPORTED_OS.include?(operating_system)
                    raise AssertionError.new("Operating system not supported: #{operating_system}")
                end

                @cuda_nvcc = cuda_manual_nvcc
                @cuda_common_include = cuda_manual_common_include
                @cuda_cupti_include = cuda_manual_cupti_include

                # Auto configuration
                if auto_config
                    Log.info("Attempting CUDA path auto configuration")

                    nvcc_path = %x(which nvcc)

                    if $?.exitstatus == 0
                        cuda_path = File.expand_path("../..", nvcc_path)
                        @cuda_nvcc = File.expand_path("bin/nvcc", cuda_path)
                        @cuda_common_include = File.expand_path("samples/common/inc", cuda_path)
                        @cuda_cupti_include = File.expand_path("extras/CUPTI/include", cuda_path)
                    else
                        Log.warn("CUDA path auto configuration failed")
                    end
                end

                # Check if nvcc is installed
                %x(#{@cuda_nvcc} 2>&1)
                if $?.exitstatus != 1
                    raise AssertionError.new("nvcc not installed")
                end

                if !File.directory?(@cuda_common_include)
                    raise AssertionError.new("Directory does not exist: #{@cuda_common_include}. Check OS configuration!")
                end

                if !File.directory?(@cuda_cupti_include)
                    raise AssertionError.new("Directory does not exist: #{@cuda_cupti_include}. Check OS configuration!")
                end

                @is_initialized = true
            end

            def nvcc_invocation_string(in_file, out_file)
                if !is_initialized?
                    reinitialize!
                end

                return "#{@cuda_nvcc} -o #{out_file} -I#{@cuda_common_include} -I#{@cuda_cupti_include} -lcurand --shared -Xcompiler -fPIC -std=c++11 #{in_file} 2>&1"
            end

            def so_suffix
                if operating_system == :linux
                    "so"
                elsif operating_system == :macosx
                    "so"
                elsif operating_system == :windows
                    "dll"
                else
                    raise AssertionError.new("Operating system not supported")
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
                        raise AssertionError.new("Unknown operating system")
                end
            end
        end
    end
end
