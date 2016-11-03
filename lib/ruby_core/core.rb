module Ikra
    module RubyIntegration
        # TODO: Handle non-singleton types

        PROMOTE_TO_FIXNUM_FLOAT = proc do |other|
            if other.singleton_type != INT.singleton_type &&
                    other.singleton_type != FLOAT.singleton_type
                raise "Expected type Fixnum or Float, found #{other}"
            else
                other
            end
        end

        ASSERT_AND_RETURN_FIXNUM = proc do |other|
            if other.singleton_type != INT.singleton_type
                raise "Expected type Fixnum, found #{other}"
            else
                INT
            end
        end

        ASSERT_NUMERIC_RETURN_BOOL = proc do |other|
            if other.singleton_type != INT.singleton_type &&
                    other.singleton_type != FLOAT.singleton_type
                raise "Expected type Fixnum or Float, found #{other}"
            else
                BOOL
            end
        end

        ASSERT_AND_RETURN_BOOL = proc do |other|
            if other.singleton_type != BOOL.singleton_type
                raise "Expected type Bool, found #{other}"
            else
                BOOL
            end
        end

        # TODO: fix int % float
        implement Fixnum, :%, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) % (#2))"
        implement Fixnum, :&, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) & (#2))"
        implement Fixnum, :|, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) | (#2))"
        implement Fixnum, :*, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) * (#2))"
        # TODO: Implement integer pow
        implement Fixnum, :**, FLOAT, 1, "powf((float) (#1), (float) (#2))"
        implement Fixnum, :+, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) + (#2))"
        implement Fixnum, :-, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) - (#2))"
        # TODO: Implement unary -
        implement Fixnum, :/, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) / (#2))"
        implement Fixnum, :<, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) < (#2))"
        implement Fixnum, :<<, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) << (#2))"
        implement Fixnum, :<=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) <= (#2))"
        # TODO: Implement <=>
        implement Fixnum, :==, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) == (#2))"
        implement Fixnum, :"!=", ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) != (#2))"
        implement Fixnum, :>, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) > (#2))"
        implement Fixnum, :>=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) >= (#2))"
        implement Fixnum, :>>, ASSERT_AND_RETURN_FIXNUM, 1, "((#1) >> (#2))"
        implement Fixnum, :"^", ASSERT_AND_RETURN_FIXNUM, 1, "((#1) ^ (#2))"

	implement Fixnum, :abs, INT, 0, "((#1) < 0 ? -(#1) : (#1))"
	implement Fixnum, :bit_length, INT, 1, "((int) (ceil(log2f((#1) < 0 ? -(#1) : (#1) + 1))))"
	implement Fixnum, :div, INT, 1, "((int) ((#1) / (#2)))"
	implement Fixnum, :even?, BOOL, 0, "((#1) % 2 == 0)"
	implement Fixnum, :fdiv, FLOAT, 1, "((#1) / ((float) (#2)))"
	implement Fixnum, :magnitude, INT, 0, "((#1) < 0 ? -(#1) : (#1))"
        implement Fixnum, :modulo, PROMOTE_TO_FIXNUM_FLOAT, 1, "((#1) % (#2))"
	implement Fixnum, :odd?, BOOL, 0, "((#1) % 2 != 0)"
	implement Fixnum, :size, INT, 0, "(sizeof(int))"
	implement Fixnum, :next, INT, 0, "((#1) + 1)"
	implement Fixnum, :succ, INT, 0, "((#1) + 1)"
	implement Fixnum, :zero?, BOOL, 0, "((#1) == 0)"
        implement Fixnum, :to_f, FLOAT, 1, "((float) (#1))"
        implement Fixnum, :to_i, INT, 1, "(#1)"

        implement Float, :%, FLOAT, 1, "fmodf((#1), ((float) (#2)))"
        implement Float, :*, FLOAT, 1, "((#1) * (#2))"
        implement Float, :**, FLOAT, 1, "powf((#1), (float) (#2))"
        implement Float, :+, FLOAT, 1, "((#1) + (#2))"
        implement Float, :-, FLOAT, 1, "((#1) - (#2))"
        # TODO: Implement unary -
        implement Float, :/, FLOAT, 1, "((#1) / (#2))"
        implement Float, :<, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) < (#2))"
        implement Float, :<=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) <= (#2))"
        # TODO: Implement <=>
        implement Float, :==, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) == (#2))"
        implement Float, :"!=", ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) != (#2))"
        implement Float, :>, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) > (#2))"
        implement Float, :>=, ASSERT_NUMERIC_RETURN_BOOL, 1, "((#1) >= (#2))"

        implement Float, :abs, FLOAT, 1, "(fabsf((#1)))"
        implement Float, :floor, FLOAT, 1, "(floorf((#1)))"
        implement Float, :to_f, FLOAT, 1, "(#1)"
        implement Float, :to_i, INT, 1, "((int) (#1))"

        implement TrueClass, :&, ASSERT_AND_RETURN_BOOL, 1, "((#1) & (#2))"
        implement TrueClass, :"!", BOOL, 1, "(!(#1))"
        implement TrueClass, :"&&", ASSERT_AND_RETURN_BOOL, 1, "((#1) && (#2))"
        implement TrueClass, :"^", ASSERT_AND_RETURN_BOOL, 1, "((#1) ^ (#2))"
        implement TrueClass, :|, ASSERT_AND_RETURN_BOOL, 1, "((#1) | (#2))"
        implement TrueClass, :"||", ASSERT_AND_RETURN_BOOL, 1, "((#1) || (#2))"

        implement FalseClass, :&, ASSERT_AND_RETURN_BOOL, 1, "((#1) & (#2))"
        implement FalseClass, :"!", BOOL, 1, "(!(#1))"
        implement FalseClass, :"&&", ASSERT_AND_RETURN_BOOL, 1, "((#1) && (#2))"
        implement FalseClass, :"^", ASSERT_AND_RETURN_BOOL, 1, "((#1) ^ (#2))"
        implement FalseClass, :|, ASSERT_AND_RETURN_BOOL, 1, "((#1) | (#2))"
        implement FalseClass, :"||", ASSERT_AND_RETURN_BOOL, 1, "((#1) || (#2))"
    end
end
