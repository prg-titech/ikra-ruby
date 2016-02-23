class Array
    def common_superclass
        class_counter = {}
        class_counter.default = 0
        
        index = 0
        each do |cls|
            while (cls != BasicObject) do
                class_counter[cls] += 1
                cls = cls.superclass
            end
            index += 1
        end
        
        smallest = Object
        class_counter.each do |cls, counter|
            if counter == size
                if cls < smallest
                    smallest = cls
                end
            end
        endl
        
        smallest
    end
    
    def all_types
        type_counter.keys
    end
    
    alias :old_push :push
    
    def push(element)
        old_push(element)
        type_counter[element.class] += 1
    end
    
    alias :old_set :[]=
    
    def []=(index, element)
        type_counter[self[index].class] -= 1
        if type_counter[self[index].class] == 0
            type_counter.delete(self[index].class)
        end
        
        old_set(index, element)
        type_counter[element.class] += 1
    end
    
    private
    
    def type_counter
        if @type_counter == nil
            @type_counter = {}
            @type_counter.default = 0
            
            each do |element|
                @type_counter[element.class] += 1
            end
        end
        
        @type_counter
    end
end
