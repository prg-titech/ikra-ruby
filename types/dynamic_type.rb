require_relative "ruby_type"

class DynamicType
    include RubyType
    
    Dynamic = self.new
    
    def to_c_type
        "Dynamic"
    end
end

