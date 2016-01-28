require "parser/current"
require "sourcify"
require "pp"
require "ripper"

# OPTIMIZATIONS
# - retrieve max. number of registers, then ?

class ArrayCommand
    def [](index)
        return result()[index]
    end
    
    def pmap(&block)
        return ArrayMapCommand.new(self, &block)
    end
    
    def pselect(&block)
        return ArraySelectCommand.new(self, &block)
    end
    
    protected
    
    def result
        @cached ||= execute()
        return @cached
    end
    
    class Translator
        def translate(source, funcName)
            ast = self.parser.parse(source)
            @functions = {funcName => ast}
            result = ""
            
            while not @functions.empty? do
                pair = @functions.first
                @functions.delete(pair[0])
                result += "function #{pair[0]}()\n" + translateBlock(pair[1])
            end
            
            return result
        end
        
        def translateBlock(node)
            if node.type != :begin
                puts "Error: BEGIN block expected"
                exit
            end
            
            content = node.children.map do |s| translateAst(s) end.join(";\n") + "\n"
            indented = content.split("\n").map do |line| "    " + line end.join("\n")
            return "{\n#{indented}\n}\n"
        end
        
        def translateAst(node)
            return case node.type
                when :int
                    "#{node.children[0].to_s}"
                when :lvar
                    node.children[0].to_s
                when :lvasgn
                    node.children[0].to_s + " = " + translateAst(node.children[1])
                when :send
                    receiver = node.children[0] == nil ? "" : translateAst(node.children[0]) + "."
                    receiver + node.children[1].to_s + "(" + node.children[2..-1].map { |c| translateAst(c) }.join(", ") + ")"
                when :begin
                    if node.children.size == 1
                        translateAst(node.children[0])
                    else
                        funcId = nextFuncId
                        @functions[funcId] = node
                        funcId + "()"
                    end
                when :if
                    result = "if (#{translateAst(node.children[0])})\n" + translateBlock(node.children[1])
                    
                    if node.children.size > 2
                        result + "else\n" + translateBlock(node.children[2])
                    else
                        result
                    end
                when :args
                    ""
                else
                    ""
            end
        end
        
        def nextId
            @nextId ||= 0
            @nextId += 1
            return @nextId
        end
        
        def nextFuncId
            return "func_" + nextId.to_s
        end
        
        def parser
            return Parser::CurrentRuby
        end
    end
end

class ArrayMapCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().map(&@block)
    end
    
    def source
        ast = self.parser.parse(@block.to_source(strip_enclosure: true))
        #ast = Ripper.sexp(@block.to_source(strip_enclosure: true))
        pp ast
        puts ArrayCommand::Translator.new.translate(@block.to_source(strip_enclosure: true), "main")
        #pp ast.children[0]
        # use @block.binding to get local variables in context
    end
    
    def size
        return @target.size
    end
    
    def parser
        return Parser::CurrentRuby
    end
end

class ArraySelectCommand < ArrayCommand
    def initialize(target, &block)
        @target = target
        @block = block
    end
    
    def execute
        # TODO: insert CUDA implementation
        return @target.execute().select(&@block)
    end
    
    def source
        puts self.parser.parse(@block.to_source(strip_enclosure: true))
    end
    
    def size
        return @target.size
    end
end

class ArrayIdentityCommand < ArrayCommand
    def initialize(target)
        @target = target
    end
    
    def execute
        return @target
    end
    
    def size
        return @target.size
    end
end

class Array
    def pmap(&block)
        return ArrayIdentityCommand.new(self).pmap(&block)
    end
end

def bla

end

#x= 1000
#p = Proc.new do
#    x+1+bla()
#end
#puts p.call()
#puts p.binding.local_variables


#puts ([1,2,3].pmap do |x| x+1 end.pmap do |x| x *10 end.pselect do |x| x > 20 end).source()
#a=[1,2,3].pmap do |x| x + 2 end
#    .pmap { |x | x=0; x * 10 * foo(1,2,3)
#    Array.new {|y| y + x} }
#a.source

a = [1,2,3].pmap do |x| 
    x + 2
    y = x + 2 * 5
    foo(begin
        puts 123
        4
    end)
    
    if x then
        puts 123
    else
        puts x
    end
    9+0
    x.foo().bar()
end
a.source