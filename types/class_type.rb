require_relative "ruby_type"
require_relative "../sourcify/lib/sourcify"
require_relative "../parsing"
require_relative "../ast/builder"

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

    def method_ast(selector)
        source = Parsing.parse_method(cls.instance_method(selector))
        Ikra::AST::Builder.from_parser_ast(source)
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