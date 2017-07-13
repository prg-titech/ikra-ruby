require "ikra"
require_relative "unit_test_template"

class RandomTest < UnitTestCase

  def test_random
    rands = Array.pnew(100) { rand }
    rands.to_a.all? { |x| x > 0 && x <= 1 }
  end

end
