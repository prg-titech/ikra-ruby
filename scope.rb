require "set"

class Frame < Hash
    def function_frame!
        @is_function_frame = true
    end

    def is_function_frame?
        @is_function_frame ||= false
        @is_function_frame
    end

    def add_return_types(types)
        if not is_function_frame?
            raise "Return types allowed only for function frames"
        end

        if types.class != Set
            raise "Expected set of types"
        end

        @return_types ||= Set.new
        @return_types.merge(types)
    end

    def return_types
        if not is_function_frame?
            raise "Return types allowed only for function frames"
        end

        @return_types ||= Set.new
        @return_types
    end
end

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
        frame = Frame.new
        frame.default_proc = Proc.new do |hash, key|
            hash[key] = Variable.new
        end
        push(frame)
    end
    
    def pop_frame
        pop
    end
    
    # TODO: maybe remove?
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
        if types.class != Set
            raise "Expected set of types (got #{types.class})"
        end

        top_frame[name].types.merge(types)
    end
    
    def add_return_types(types)
        reverse_each.detect do |fr|
            fr.add_return_types(types)
            return self
        end

        raise "Function frame not found"
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