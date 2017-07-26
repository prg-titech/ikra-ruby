require "ikra"
require_relative "unit_test_template"

class RandomTest < UnitTestCase

    def test_rand_between_zero_and_one
        rands = Array.pnew(100) { rand }
        assert rands.to_a.all? { |x| x > 0 && x <= 1 }
    end

    def test_rand_not_equal_in_different_threads
        rands = Array.pnew(2) { rand }
        assert_not_equal rands[0], rands[1]
    end

    def test_rand_not_equal_in_the_same_thread
        not_equal = Array.pnew(1) do
            a = rand
            b = rand
            return a != b
        end
        assert not_equal[0]
    end

    def test_rand_equal_after_srand
        equal = Array.pnew(1) do
            srand 0
            a = rand
            srand 0
            b = rand
            return a == b
        end
        assert equal[0]
    end

end
