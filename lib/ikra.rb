require "logger"

module Ikra
    Log = Logger.new(STDOUT)

    class AssertionError < RuntimeError

    end
end

require_relative "ruby_core/ruby_integration"
require_relative "symbolic/symbolic"
require_relative "entity"
require_relative "translator/cuda_errors"
require_relative "cpu/cpu_implementation"