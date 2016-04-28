require_relative "../ikra"

class A
	def max(a, b)
		if a > b
			a
		else
			b
		end
	end

	class << self
		def static_max(a, b)
			if a > b
				a
			else
				b
			end
		end
	end
end

a = A.new

result = Array.pnew(10) do |i|
	x1 = i * i
	x2 = 2 * i

	res = a.max(x1, x2)
	#result2 = A.static_max(x1, x2)
	res
end

sum = 0
(0..result.size - 1).each do |index|
    sum += result[index]
end

puts sum
