module Ikra
    module RubyIntegration
        # TODO: Handle non-singleton types

        TYPE_INT_COERCE_TO_FLOAT = proc do |recv, other|
            if other.include?(FLOAT_S) 
                FLOAT
            elsif other.include?(INT_S)
                INT
            else
                # At least one of the types INT_S or FLOAT_S are required
                raise "Operation defined numeric values only (found #{other})"
            end
        end

        TYPE_INT_RETURN_INT = proc do |recv, other|
            if !other.include?(INT_S)
                raise "Operation defined Int values only (found #{other})"
            end

            INT
        end

        TYPE_NUMERIC_RETURN_BOOL = proc do |recv, other|
            if !other.include?(INT_S) && !other.include?(FLOAT_S)
                raise "Expected type Int or Float, found #{other}"
            end

            BOOL
        end

        TYPE_BOOL_RETURN_BOOL = proc do |recv, other|
            if !other.include?(BOOL_S)
                raise "Expected type Bool, found #{other}"
            end

            BOOL
        end

        # TODO: fix int % float
        implement INT_S, :%, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 % #N1)"
        implement INT_S, :&, TYPE_INT_RETURN_INT, 1, "(#0 & #I1)"
        implement INT_S, :|, TYPE_INT_RETURN_INT, 1, "(#0 | #I1)"
        implement INT_S, :*, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 * #N1)"
        # TODO: Find better implementation for Int pow
        implement INT_S, :**, INT, 1, "((int) pow((double) #0, (double) #F1))"
        implement INT_S, :+, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 + #N1)"
        implement INT_S, :-, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 - #N1)"
        # TODO: Implement unary -
        implement INT_S, :/, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 / #N1)"
        implement INT_S, :<, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 < #N1)"
        implement INT_S, :<<, TYPE_INT_RETURN_INT, 1, "(#0 << #I1)"
        implement INT_S, :<=, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 <= #N1)"
        # TODO: Implement <=>
        implement INT_S, :==, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 == #N1)"
        implement INT_S, :"!=", TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 != #N1)"
        implement INT_S, :>, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 > #N1)"
        implement INT_S, :>=, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 >= #N1)"
        implement INT_S, :>>, TYPE_INT_RETURN_INT, 1, "(#0 >> #I1)"
        implement INT_S, :"^", TYPE_INT_RETURN_INT, 1, "(#0 ^ IN1)"

        implement INT_S, :abs, INT, 0, "({ int value = #0; value < 0 ? -value : value; })"
        implement INT_S, :bit_length, INT, 1, "({ int value = #0; (int) ceil(log2f(value < 0 ? -value : value + 1)); })"
        implement INT_S, :div, INT, 1, "((int) (#0 / #I1))"
        implement INT_S, :even?, BOOL, 0, "(#0 % 2 == 0)"
        implement INT_S, :fdiv, FLOAT, 1, "(#0 / #F1)"
        implement INT_S, :magnitude, INT, 0, "({ int value = #0; value < 0 ? -value : value; })"
        implement INT_S, :modulo, TYPE_INT_COERCE_TO_FLOAT, 1, "(#0 % #N1)"
        implement INT_S, :odd?, BOOL, 0, "(#0 % 2 != 0)"
        implement INT_S, :size, INT, 0, "sizeof(int)"
        implement INT_S, :next, INT, 0, "(#0 + 1)"
        implement INT_S, :succ, INT, 0, "(#0 + 1)"
        implement INT_S, :zero?, BOOL, 0, "(#0 == 0)"
        implement INT_S, :to_f, FLOAT, 1, "((float) #0)"
        implement INT_S, :to_i, INT, 1, "#0"

        implement FLOAT_S, :%, FLOAT, 1, "fmodf(#0, #F1)"
        implement FLOAT_S, :*, FLOAT, 1, "(#0 * #N1)"
        implement FLOAT_S, :**, FLOAT, 1, "powf(#0, #F1)"
        implement FLOAT_S, :+, FLOAT, 1, "(#0 + #N1)"
        implement FLOAT_S, :-, FLOAT, 1, "(#0 - #N1)"
        # TODO: Implement unary -
        implement FLOAT_S, :/, FLOAT, 1, "(#0 / #N1)"
        implement FLOAT_S, :<, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 < #N1)"
        implement FLOAT_S, :<=, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 <= #N1)"
        # TODO: Implement <=>
        implement FLOAT_S, :==, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 == #N1)"
        implement FLOAT_S, :"!=", TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 != #N1)"
        implement FLOAT_S, :>, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 > #N1)"
        implement FLOAT_S, :>=, TYPE_NUMERIC_RETURN_BOOL, 1, "(#0 >= #N1)"

        implement FLOAT_S, :abs, FLOAT, 1, "fabsf(#0)"
        implement FLOAT_S, :floor, FLOAT, 1, "floorf(#0)"
        implement FLOAT_S, :to_f, FLOAT, 1, "#0"
        implement FLOAT_S, :to_i, INT, 1, "((int) #0)"

        implement BOOL_S, :&, TYPE_BOOL_RETURN_BOOL, 1, "(#0 & #B1)"
        implement BOOL_S, :"&&", TYPE_BOOL_RETURN_BOOL, 1, "(#0 && #B1)"
        implement BOOL_S, :"^", TYPE_BOOL_RETURN_BOOL, 1, "(#0 ^ #B1)"
        implement BOOL_S, :|, TYPE_BOOL_RETURN_BOOL, 1, "(#0 | #B1)"
        implement BOOL_S, :"||", TYPE_BOOL_RETURN_BOOL, 1, "(#0 || #B1)"
    end
end
