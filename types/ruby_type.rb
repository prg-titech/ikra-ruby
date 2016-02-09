module RubyType
    def to_ruby_type
        raise NotImplementedError
    end
    
    def to_c_type
        raise NotImplementedError
    end
    
    def is_primitive?
        false
    end
end