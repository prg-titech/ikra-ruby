class Array 
    def stencil(neighbors, out_of_range_value, &block)
        copy = self.dup

        return Array.new(size) do |index|
            values = neighbors.map do |offset|
                if index + offset < 0 or index + offset >= size
                    out_of_range_value
                else
                    copy[index + offset]
                end
            end

            block.call(*values)
        end
    end
end
