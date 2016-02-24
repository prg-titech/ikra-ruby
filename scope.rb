require "set"

class Scope < Array
    class Variable
        attr_reader :types
        attr_accessor :read
        attr_accessor :written
        
        def initialize(types = Set.new)
            @types = types
            @read = false
            @written = false
        end
    end
    
    def top_frame
        last
    end
    
    def push_frame
        frame = Hash.new
        frame.default_proc = Proc.new do |hash, key|
            hash[key] = Variable.new
        end
        push(frame)
    end
    
    def pop_frame
        pop
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
    
    def get_types(name)
        reverse_each do |frame|
            if frame.has_key?(name)
                return frame[name].types
            end
        end
        
        raise "#{name} not found in symbol table"
    end
    
    def add_types(name, types)
        top_frame[name].types.merge(types)
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