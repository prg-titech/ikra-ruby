require "set"
require_relative "types/union_type"

class Frame < Hash
    def function_frame!
        @is_function_frame = true
    end

    def is_function_frame?
        @is_function_frame ||= false
        @is_function_frame
    end

    def add_return_type(type)
        if not is_function_frame?
            raise "Return type allowed only for function frames"
        end

        if not type.is_union_type?
            raise "Expected union type"
        end

        @return_type ||= Ikra::Types::UnionType.new
        @return_type.expand(type)
    end

    def return_type
        if not is_function_frame?
            raise "Return type allowed only for function frames"
        end

        @return_type ||= Ikra::Types::UnionType.new
        @return_type
    end

    def variable_names
        keys
    end
end

class Scope < Array
    class Variable
        attr_reader :type
        attr_accessor :read
        attr_accessor :written
        
        def initialize(type = Ikra::Types::UnionType.new)
            @type = type
            @read = false
            @written = false
        end
    end
    
    def top_frame
        last
    end
    
    def previous_frame
        self[-2]
    end

    def push_frame
        frame = Frame.new
        frame.default_proc = Proc.new do |hash, key|
            hash[key] = Variable.new
        end
        push(frame)
    end

    def push_function_frame
        push_frame
        top_frame.function_frame!
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
    
    def get_type(name)
        reverse_each do |frame|
            if frame.has_key?(name)
                return frame[name].type
            end
        end
        
        raise "#{name} not found in symbol table"
    end
    
    def declare_expand_type(name, type)
        if not type.is_union_type?
            raise "Expected union type"
        end

        top_frame[name].type.expand(type)
    end
    
    def add_return_type(type)
        reverse_each.detect do |fr|
            fr.add_return_type(type)
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
    
    def read_and_written_variables(frame_position = -1)
        read_variables(frame_position) + written_variables(frame_position)
    end
    
    def new_frame(&block)
        push_frame
        yield
        pop_frame
    end

    def new_function_frame(&block)
        push_function_frame
        yield
        pop_frame
    end
end