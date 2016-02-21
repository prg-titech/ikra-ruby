require_relative "../symbolic.rb"

class Vehicle

end

class Car < Vehicle

end

class Bus < Vehicle

end

class Motorcycle < Vehicle

end

arr = []
100.times do
    arr.push(Car.new)
end
100.times do
    arr.push(Bus.new)
end
500.times do
    arr.push(Motorcycle.new)
end
arr.shuffle!

# TODO: how do we know what kind of elements are in the array --> use subclass of array or modify []=, push
# TODO: how do we know what the instance variables of an object are --> 
#       1. Let programmer declare them explicitly 
#       2. Do a scan during before compilation
#       3. Use a map/hash table/two arrays in the CUDA code
#       4. Combination of 1 and 3 or undeclared variables
