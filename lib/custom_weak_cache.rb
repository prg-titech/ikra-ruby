require "weakref"

class CustomWeakCache
    def initialize(comparator_selector: :==)
        @values = []
        @comparator = comparator_selector
    end

    def get_value(value)
        @values.delete_if do |obj|
            begin
                if are_equal?(obj, value)
                    return obj.__getobj__
                end
            rescue WeakRef::RefError
                true
            end
        end

        raise RuntimeError.new("Value not found")
    end

    def add_value(value)
        @values.push(WeakRef.new(value))
    end

    def include?(value)
        @values.delete_if do |obj|
            begin
                if are_equal?(obj, value)
                    return true
                end
            rescue WeakRef::RefError
                true
            end
        end

        return false
    end

    protected

    def are_equal?(a, b)
        return a.send(@comparator, b)
    end
end

class CustomWeakHash
    class Item
        attr_reader :key
        attr_reader :value

        def initialize(key, value)
            @key = key
            @value = value
        end
    end

    def initialize(comparator_selector: :==)
        @items = []
        @comparator = comparator_selector
    end

    def [](key)
        @items.delete_if do |obj|
            begin
                if are_equal?(obj.key.__getobj__, key)
                    return obj.value
                end
            rescue WeakRef::RefError
                true
            end
        end

        raise RuntimeError.new("Key not found")
    end

    def []=(key, value)
        @items.push(Item.new(WeakRef.new(key), value))
    end

    def include?(key)
        @items.delete_if do |obj|
            begin
                if are_equal?(obj.key.__getobj__, key)
                    return true
                end
            rescue WeakRef::RefError
                true
            end
        end

        return false
    end

    protected

    def are_equal?(a, b)
        return a.send(@comparator, b)
    end
end
