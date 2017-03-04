class Array 
    def stencil(neighbors, out_of_range_value, use_parameter_array: true, with_index: false, &block)
        copy = self.dup

        return Array.new(size) do |index|
            if neighbors.min + index < 0 || neighbors.max + index > size - 1
                out_of_range_value
            else
                values = neighbors.map do |offset|
                    copy[index + offset]
                end

                if use_parameter_array
                    if with_index
                        block.call(values, index)
                    else
                        block.call(values)
                    end
                else
                    if with_index
                        block.call(*values, index)
                    else
                        block.call(*values)
                    end
                end
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
