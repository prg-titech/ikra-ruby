module Ikra
    module RubyIntegration
        # TODO: Handle non-singleton types

        PROMOTE_TO_FIXNUM_FLOAT = proc do |recv, other|
            if other.singleton_type != INT.singleton_type &&
                    other.singleton_type != FLOAT.singleton_type
                raise "Expected type Int or Float, found #{other}"
            else
                other
            end
        end

        ASSERT_AND_RETURN_FIXNUM = proc do |recv, other|
            if other.singleton_type != INT.singleton_type
                raise "Expected type Int, found #{other}"
            else
                INT
            end
        end

        ASSERT_NUMERIC_RETURN_BOOL = proc do |recv, other|
            if other.singleton_type != INT.singleton_type &&
                    other.singleton_type != FLOAT.singleton_type
                raise "Expected type Int or Float, found #{other}"
            else
                BOOL
            end
        end

        ASSERT_AND_RETURN_BOOL = proc do |recv, other|
            if other.singleton_type != BOOL.singleton_type
                raise "Expected type Bool, found #{other}"
            else
                BOOL
            end
        end

        INT_S = INT.singleton_type
        FLOAT_S = FLOAT.singleton_type
        BOOL_S = BOOL.singleton_type

        # TODO: fix int % float
        implement INT_S, :%, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) % (#2))"
        implement INT_S, :&, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) & (#2))"
        implement INT_S, :|, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) | (#2))"
        implement INT_S, :*, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) * (#2))"
        # TODO: Implement integer pow
        implement INT_S, :**, FLOAT, 1, "powf((float) (#1), (float) (#2))"
        implement INT_S, :+, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) + (#2))"
        implement INT_S, :-, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) - (#2))"
        # TODO: Implement unary -
        implement INT_S, :/, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) / (#2))"
        implement INT_S, :<, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) < (#2))"
        implement INT_S, :<<, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) << (#2))"
        implement INT_S, :<=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) <= (#2))"
        # TODO: Implement <=>
        implement INT_S, :==, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) == (#2))"
        implement INT_S, :"!=", ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) != (#2))"
        implement INT_S, :>, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) > (#2))"
        implement INT_S, :>=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) >= (#2))"
        implement INT_S, :>>, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) >> (#2))"
        implement INT_S, :"^", ASSERT_AND_RETURN_FIXNUM, 1, "((#1) ^ (#2))"

        implement INT_S, :abs, INT, 0, "((#1) < 0 ? -(#1) : (#1))"
        implement INT_S, :bit_length, INT, 1, "((int) (ceil(log2f((#1) < 0 ? -(#1) : (#1) + 1))))"
        implement INT_S, :div, INT, 1, "((int) ((#1) / (#2)))"
        implement INT_S, :even?, BOOL, 0, "((#1) % 2 == 0)"
        implement INT_S, :fdiv, FLOAT, 1, "((#1) / ((float) (#2)))"
        implement INT_S, :magnitude, INT, 0, "((#1) < 0 ? -(#1) : (#1))"
        implement INT_S, :modulo, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) % (#2))"
        implement INT_S, :odd?, BOOL, 0, "((#1) % 2 != 0)"
        implement INT_S, :size, INT, 0, "(sizeof(int))"
        implement INT_S, :next, INT, 0, "((#1) + 1)"
        implement INT_S, :succ, INT, 0, "((#1) + 1)"
        implement INT_S, :zero?, BOOL, 0, "((#1) == 0)"
        implement INT_S, :to_f, FLOAT, 1, "((float) (#1))"
        implement INT_S, :to_i, INT, 1, "(#1)"

        implement FLOAT_S, :%, FLOAT, 1, "fmodf((#1), ((float) (#2)))"
        implement FLOAT_S, :*, FLOAT, 1, "((#1) * (#2))"
        implement FLOAT_S, :**, FLOAT, 1, "powf((#1), (float) (#2))"
        implement FLOAT_S, :+, FLOAT, 1, "((#1) + (#2))"
        implement FLOAT_S, :-, FLOAT, 1, "((#1) - (#2))"
        # TODO: Implement unary -
        implement FLOAT_S, :/, FLOAT, 1, "((#1) / (#2))"
        implement FLOAT_S, :<, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) < (#2))"
        implement FLOAT_S, :<=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) <= (#2))"
        # TODO: Implement <=>
        implement FLOAT_S, :==, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) == (#2))"
        implement FLOAT_S, :"!=", ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) != (#2))"
        implement FLOAT_S, :>, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) > (#2))"
        implement FLOAT_S, :>=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) >= (#2))"

        implement FLOAT_S, :abs, FLOAT, 1, "(fabsf((#1)))"
        implement FLOAT_S, :floor, FLOAT, 1, "(floorf((#1)))"
        implement FLOAT_S, :to_f, FLOAT, 1, "(#1)"
        implement FLOAT_S, :to_i, INT, 1, "((int) (#1))"

        implement BOOL_S, :&, ASSERT_AND_RETURN_BOOL, 1, "((#1) & (#2))"
        implement BOOL_S, :"&&", ASSERT_AND_RETURN_BOOL, 1, "((#1) && (#2))"
        implement BOOL_S, :"^", ASSERT_AND_RETURN_BOOL, 1, "((#1) ^ (#2))"
        implement BOOL_S, :|, ASSERT_AND_RETURN_BOOL, 1, "((#1) | (#2))"
        implement BOOL_S, :"||", ASSERT_AND_RETURN_BOOL, 1, "((#1) || (#2))"
    end
end
