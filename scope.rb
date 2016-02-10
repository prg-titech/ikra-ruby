class Scope < Array
    class Variable
        attr_reader :type
        attr_accessor :read
        attr_accessor :written
        
        def initialize(type)
            @type = type
            @read = false
            @written = false
        end
    end
    
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
        
        top_frame[name] = Variable.new(type)
    end
    
    def define_shadowed(name, type)
        if top_frame.has_key?(name)
            raise "#{name} already defined"
        end
        
        top_frame[name] = Variable.new(type)
    end
    
    def is_defined?(name)
        any? do |frame|
            frame.has_key?(name)
        end
    end
    
    def get_type(name)
        reverse_each do |frame|
            if frame.has_key?(name)
                return frame[name].type
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
    
    def read!(name)
        frame = reverse_each.detect do |fr|
            fr.has_key?(name)
        end
        
        frame[name].read = true
    end
    
    def written!(name)
        frame = reverse_each.detect do |fr|
            fr.has_key?(name)
        end
        
        frame[name].written = true
    end
    
    def read_variables(frame_position)
        frame = self[frame_position]
        frame.select do |name, var|
            var.read
        end.keys
    end
    
    def written_variables(frame_position)
        frame = self[frame_position]
        frame.select do |name, var|
            var.written
        end.keys
    end
    
    def read_and_written_variables(frame_position)
        read_variables(frame_position) + written_variables(frame_position)
    end
    
    def new_frame(&block)
        push_frame
        yield
        pop_frame
    end
end