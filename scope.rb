class Scope < Array
    def top_frame
        last
    end
    
    def push_frame
        push(Hash.new)
    end
    
    def pop_frame
        pop
    end
    
    def define(name, type)
        if is_defined?(name)
            raise "#{name} already defined"
        end
        
        top_frame[name] = type
    end
    
    def define_shadowed(name, type)
        if top_frame.has_key?(name)
            raise "#{name} already defined"
        end
        
        top_frame[name] = type
    end
    
    def is_defined?(name)
        any? do |frame|
            frame.has_key?(name)
        end
    end
    
    def get_type(name)
        each do |frame|
            if frame.has_key?(name)
                return frame[name]
            end
        end
        
        raise "#{name} not found in symbol table"
    end
    
    def ensure_defined(name, type)
        if is_defined?(name) and get_type(name) != type
            raise "Expected that #{name} has type #{type} but found #{get_type(name)}"
        elsif not is_defined?(name)
            define(name, type)
        end
    end
    
    def new_frame(&block)
        push_frame
        yield
        pop_frame
    end
end