require "parser/current"
require_relative "sourcify/lib/sourcify"

module Parsing
    class << self
        def parse_block(block, local_variables = [])
            parser = Parser::CurrentRuby.default_parser
            local_variables.each do |var|
                parser.static_env.declare(var)
            end
            
            parser_source = Parser::Source::Buffer.new('(string)', 1)
            parser_source.source = block.to_source(strip_enclosure: true)
            
            return parser.parse(parser_source)
        end
        
        def parse_method(method)
            parser = Parser::CurrentRuby.default_parser
            method.parameters.each do |param|
                parser.static_env.declare(param[1])
            end
            
            parser_source = Parser::Source::Buffer.new('(string)', 1)
            parser_source.source = method.to_source(strip_enclosure: true)
            
            return parser.parse(parser_source)
        end
    end
end