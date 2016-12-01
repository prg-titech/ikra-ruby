class Array 
    def stencil(neighbors, out_of_range_value, use_parameter_array: true, &block)
        copy = self.dup

        return Array.new(size) do |index|
            values = neighbors.map do |offset|
                if index + offset < 0 or index + offset >= size
                    out_of_range_value
                else
                    copy[index + offset]
                end
            end

            if use_parameter_array
                block.call(values)
            else
                block.call(*values)
            end
        end
    end

    def combine(*others, &block)
        return Array.new(self.size) do |index|
            other_elements = others.map do |other|
                other[index]
            end

            block.call(self[index], *other_elements)
        end


    end
end
