require_relative "../ikra"

array = (1..10000).to_a
squared = array.pmap do |value|
    value * value
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
