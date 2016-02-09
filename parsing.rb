require "parser/current"
require "sourcify"

module Parsing
    class << self
        def parse(block, local_variables = [])
            parser = Parser::CurrentRuby.default_parser
            local_variables.each do |var|
                parser.static_env.declare(var)
            end
            
            parser_source = Parser::Source::Buffer.new('(string)', 1)
            parser_source.source = block.to_source(strip_enclosure: true)
            
            return parser.parse(parser_source)
        end
    end
end