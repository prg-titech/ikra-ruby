require_relative "../ikra"

class MyObject1
    include Ikra::Entity

    def initialize(value)
        @value = value
    end

    def squared
        @value * @value
    end

    def to_s
        "MyObject1 instance"
    end
end

class MyObject2
    include Ikra::Entity

    def initialize(value)
        @value = value * 2
    end

    def squared
        (@value / 2) * (@value / 2)
    end

    def to_s
        "MyObject2 instance"
    end

end

base = []
for i in 1..10000
    if i % 2 == 0
        base.push(MyObject1.new(i))
    else
        base.push(MyObject2.new(i))
    end
end

squared = base.pmap do |el|
    el.squared
end


sum = 0
(0..squared.size - 1).each do |index|
    sum += squared[index]
end

if sum == 333383335000
    puts "Test passed!"
else
    puts "Test failed: expected 333383335000 but found #{sum}"
end

