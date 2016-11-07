module Ikra
    module Errors
        class CudaError < Exception

        end

        class CudaErrorIllegalAddress < CudaError
        
        end

        class CudaUnknownError < CudaError
            attr_reader :error_code

            def initialize(error_code)
                @error_code = error_code
            end

            def to_s
                "CudaUnknownError (#{error_code})"
            end
        end

        def self.raiseCudaError(error_code)
            case error_code
            when 77
                raise CudaErrorIllegalAddress.new
            else
                raise CudaUnknownError.new(error_code)
            end
        end
    end
end