require_relative "ruby_type"

class ClassType
    include RubyType

    attr_reader :cls

    class << self
        def new(cls)
            if @cache == nil
                @cache = {}
                @cache.default_proc = Proc.new do |hash, key|
                    hash[key] = super(key)
                end
            end

            @cache
        end
    end

    def initialize(cls)
        @cls = cls
    end

    def to_s
        "<class: #{@cls.to_s}>"
    end
end

class Object
    def self.to_ikra_type
        ClassType.new(self)
    end
end