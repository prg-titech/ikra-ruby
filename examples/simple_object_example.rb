require_relative "../ikra"

class MyObject
    include Ikra::Entity

    def initialize(value)
        @value = value
    end

    def squared
        @value * @value
    end
end

base = []
for i in 1..10000
    base.push(MyObject.new(i))
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

