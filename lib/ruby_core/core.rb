module Ikra
    module RubyIntegration
        # TODO: Handle non-singleton types

        TYPE_INT_RETURN_ALL_NUMERIC = proc do |recv, other|
            return_type = Types::UnionType.new

            if other.include?(INT_S) 
                return_type.add(INT_S)
            end

            if other.include?(FLOAT_S)
                return_type.add(FLOAT_S)
            end

            if return_type.empty?
                # At least one of the types INT_S or FLOAT_S are required
                raise "Operation defined numeric values only (found #{other})"
            end

            return_type
        end

        def self.int_to_num_code_operator(operator)
            return code_operator(operator, TYPE_INT_RETURN_ALL_NUMERIC)
        end

        # Generates code for a numeric operator where:
        # 1. Receiver is of type [recv_type]
        # 2. Operand is either an Int or a Float, or a union type of both (if [allow_float])
        # 3. If operand is another type at runtime, a runtime exception should be thrown (TODO)
        # 4. Return value is a union type if [allow_float] and [recv_type] is Int
        # 5. The runtime return value is determined/promoted by runtime type of operand if 
        #    the receiver type is Int
        def self.code_operator(operator, return_type)
            # Receiver is guaranteed to be a singleton Int, but other can be any type

            return proc do |args_types, args_code|
                recv_type = args_types[0]
                other_type = args_types[1]

                if return_type.is_a?(Proc)
                    return_type = return_type.call(args_types[0], other_type)
                end

                puts return_type
                
                if other_type.is_singleton?
                    # Type is guaranteed to be (either) Int or Float (otherwise type inference 
                    # would have failed earlier)
                    "(#0 #{operator} #1)"
                else
                    # Perform a type check, then generate union type value

                    return_value_is_union_type = !return_type.is_singleton?

                    result = StringIO.new
                    result << "({ #{recv_type.to_c_type} _op1 = #0;\n"
                    result << "union_t _op2 = #1;\n"

                    if return_value_is_union_type
                        # In this case, the type of the operand determines the return type
                        # at runtime. We cannot determine the type statically.
                        result << "union_t _result;\n"
                    else
                        # Explicitly specified return type
                        result << "#{return_type.to_c_type} _result;\n"
                    end

                    result << "switch (_op2.class_id) {\n"

                    # ---- OPERAND IS INT ----
                    result << "    case #{INT_S.class_id}:\n"

                    if return_value_is_union_type
                        if recv_type == INT_S
                            # INT x INT --> INT
                            result << "        _result = ((union_t) {#{INT_S.class_id}, {.int_ = _op1 #{operator} _op2.value.int_}});\n"
                        elsif recv_type == FLOAT_S
                            # FLOAT x INT --> FLOAT
                            result << "        _result = ((union_t) {#{FLOAT_S.class_id}, {.float_ = _op1 #{operator} _op2.value.int_}});\n"
                        end
                    else
                        result << "        _result = _op1 #{operator} _op2.value.int_;\n"
                    end

                    result << "        break;\n"

                    if other_type.include?(FLOAT_S)
                        # ---- OPERAND IS FLOAT ----
                        result << "    case #{FLOAT_S.class_id}:\n"

                        if return_value_is_union_type
                            result << "        _result = ((union_t) {#{FLOAT_S.class_id}, {.float_ = _op1 #{operator} _op2.value.float_}});\n"
                        else
                            result << "        _result = _op1 #{operator} _op2.value.float_;\n"
                        end
                    end

                    result << "        break;\n"

                    result << "    default:\n"
                    # TODO: Set error state
                    result << "        // TODO: error handling\n"
                    result << "        break;\n"

                    result << "}\n"
                    result << "_result; })"

                    result.string
                end
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
        implement INT_S, :%, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("%")
        implement INT_S, :&, TYPE_INT_RETURN_INT, 1, code_operator("&", INT)
        implement INT_S, :|, TYPE_INT_RETURN_INT, 1, code_operator("|", INT)
        implement INT_S, :*, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("*")
        # TODO: Find better implementation for Int pow
        implement INT_S, :**, INT, 1, "((int) pow((double) #0, (double) #F1))"
        implement INT_S, :+, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("+")
        implement INT_S, :-, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("-")
        # TODO: Implement unary -
        implement INT_S, :/, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("/")
        implement INT_S, :<, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("<", BOOL)
        implement INT_S, :<<, TYPE_INT_RETURN_INT, 1, code_operator("<<", INT)
        implement INT_S, :<=, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("<=", BOOL)
        # TODO: Implement <=>
        implement INT_S, :==, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("==", BOOL)
        implement INT_S, :"!=", TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("!=", BOOL)
        implement INT_S, :>, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator(">", BOOL)
        implement INT_S, :>=, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator(">=", BOOL)
        implement INT_S, :>>, TYPE_INT_RETURN_INT, 1, code_operator(">>", INT)
        implement INT_S, :"^", TYPE_INT_RETURN_INT, 1, code_operator("^", INT)

        implement INT_S, :abs, INT, 0, "({ int value = #0; value < 0 ? -value : value; })"
        implement INT_S, :bit_length, INT, 1, "({ int value = #0; (int) ceil(log2f(value < 0 ? -value : value + 1)); })"
        implement INT_S, :div, INT, 1, "((int) (#0 / #I1))"
        implement INT_S, :even?, BOOL, 0, "(#0 % 2 == 0)"
        implement INT_S, :fdiv, FLOAT, 1, "(#0 / #F1)"
        implement INT_S, :magnitude, INT, 0, "({ int value = #0; value < 0 ? -value : value; })"
        implement INT_S, :modulo, TYPE_INT_RETURN_ALL_NUMERIC, 1, int_to_num_code_operator("%")
        implement INT_S, :odd?, BOOL, 0, "(#0 % 2 != 0)"
        implement INT_S, :size, INT, 0, "sizeof(int)"
        implement INT_S, :next, INT, 0, "(#0 + 1)"
        implement INT_S, :succ, INT, 0, "(#0 + 1)"
        implement INT_S, :zero?, BOOL, 0, "(#0 == 0)"
        implement INT_S, :to_f, FLOAT, 1, "((float) #0)"
        implement INT_S, :to_i, INT, 1, "#0"

        implement FLOAT_S, :%, FLOAT, 1, "fmodf(#0, #F1)"
        implement FLOAT_S, :*, FLOAT, 1, code_operator("*", FLOAT)
        implement FLOAT_S, :**, FLOAT, 1, "powf(#0, #F1)"
        implement FLOAT_S, :+, FLOAT, 1, code_operator("+", FLOAT)
        implement FLOAT_S, :-, FLOAT, 1, code_operator("-", FLOAT)
        # TODO: Implement unary -
        implement FLOAT_S, :/, FLOAT, 1, code_operator("/", FLOAT)
        implement FLOAT_S, :<, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("<", BOOL)
        implement FLOAT_S, :<=, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("<=", BOOL)
        # TODO: Implement <=>
        implement FLOAT_S, :==, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("==", BOOL)
        implement FLOAT_S, :"!=", TYPE_NUMERIC_RETURN_BOOL, 1, code_operator("!=", BOOL)
        implement FLOAT_S, :>, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator(">", BOOL)
        implement FLOAT_S, :>=, TYPE_NUMERIC_RETURN_BOOL, 1, code_operator(">=", BOOL)

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
