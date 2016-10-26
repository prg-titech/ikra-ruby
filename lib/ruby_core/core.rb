module Ikra
    module RubyIntegration
        implement Fixnum, :to_f, :float, 1, "(float) #1"
        implement Fixnum, :to_i, :int, 1, "#1"

        implement Float, :to_f, :float, 1, "#1"
        implement Float, :to_i, :int, 1, "(int) #1"
    end
end
