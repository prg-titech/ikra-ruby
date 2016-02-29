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

class Array
	def to_type_array_string
		"[" + map do |set|
			"{" + set.map do |type|
				type.to_s
			end.join(", ") + "}"
		end.join(", ") + "]"
	end
end
