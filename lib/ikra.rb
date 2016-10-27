require "logger"

module Ikra
    Log = Logger.new(STDOUT)
end

require_relative "ruby_core/ruby_integration"
require_relative "symbolic/symbolic"
require_relative "entity"
