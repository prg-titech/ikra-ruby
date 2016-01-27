class ArrayCommand
    def [](index)
        return result()[index]
    end
    
    def result
        @cached ||= execute()
        return @cached
    end
    
    def pmap(&block)
        return ArrayMapCommand.new(self, &block)
    end
    
    def pselect(&block)
        return ArraySelectCommand.new(self, &block)
    end
end

class ArrayMapCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().map(&@block)
    end
end

class ArraySelectCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().select(&@block)
    end
end

class ArrayIdentityCommand < ArrayCommand
    def initialize(target)
        @target = target
    end
    
    def execute
        return @target
    end
end

class Array
    def pmap(&block)
        return ArrayIdentityCommand.new(self).pmap(&block)
    end
end

puts ([1,2,3].pmap do |x| x+1 end.pmap do |x| x *10 end.pselect do |x| x > 20 end) [0]