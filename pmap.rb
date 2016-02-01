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
        VAR_DECL_SYMBOL = :DECL_ONLY
        
        def translate(source, funcName, externs={})
            @symbolTable = SymbolTable.new
            @symbolTable.merge!(externs)
            
            modifiedSource = externs.keys.reduce("") do |acc, n|
                acc + n.to_s + " = :" + VAR_DECL_SYMBOL.to_s + "\n"
            end + source
            
            ast = self.parser.parse(modifiedSource)
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
                raise "ERROR: BEGIN block expected"
            end
            
            content = node.children.map do |s| translateAst(s) end.join(";\n") + "\n"
            indented = content.split("\n").map do |line| "    " + line end.join("\n")
            return "{\n#{indented}\n}\n"
        end
        
        BINARY_ARITHMETIC_OPERATORS = [:+, :-, :*, :/]
        BINARY_COMPARISON_OPERATORS = [:==, :>, :<, :>=, :<=, :!=]
        BINARY_LOGICAL_OPERATORS = [:|, :"||", :&, :"&&", :^]
        
        class TranslationResult
            def initialize(translation, type)
                @translation = translation
                @type = type
            end
            
            def translation
                return @translation
            end
            
            def type
                return @type
            end
            
            def to_str
                return translation
            end
            
            def to_s
                return to_str
            end
            
            def +(other)
                return to_str + other.to_str
            end
        end
        
        def translateAst(node)
            return case node.type
                when :int
                    TranslationResult.new(node.children[0].to_s, SymbolTable::PrimitiveType::INT)
                when :float
                    TranslationResult.new(node.children[0].to_s, SymbolTable::PrimitiveType::FLOAT)
                when :lvar
                    TranslationResult.new(node.children[0].to_s, @symbolTable[node.children[0].to_sym])
                when :lvasgn
                    if (node.children[1].type == :sym && node.children[1].children[0] == Translator::VAR_DECL_SYMBOL)
                        # TODO: find better way to do this
                        # TODO: I think we do not need this because this is part of the method signature
                        # This is only a variable declaration without an assignment
                        TranslationResult.new(@symbolTable[node.children[0].to_sym].cTypeName + " " + node.children[0].to_s + ";", SymbolTable::PrimitiveType::VOID)
                    else
                        value = translateAst(node.children[1])
                        
                        if (@symbolTable.has_key?(node.children[0].to_sym))
                            @symbolTable.assert(node.children[0].to_sym, value.type)
                            TranslationResult.new(node.children[0].to_s + " = " + value, value.type)
                        else
                            @symbolTable.insert(node.children[0].to_sym, value.type)
                            TranslationResult.new(value.type.cTypeName + " " + node.children[0].to_s + " = " + value, value.type)
                        end
                    end
                when :send
                    receiver = node.children[0] == nil ? TranslationResult.new("", SymbolTable::PrimitiveType::VOID) : translateAst(node.children[0])
                    
                    if BINARY_ARITHMETIC_OPERATORS.include?(node.children[1])
                        operand = translateAst(node.children[2])
                        value = "(" + receiver + " " + node.children[1].to_s + " " + operand + ")"
                        
                        if receiver.type.isPrimitive and operand.type.isPrimitive
                            # implicit type conversions allowed
                            
                            if [receiver.type, operand.type].include?(SymbolTable::PrimitiveType::BOOL)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL"
                            elsif [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::VOID)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to VOID"
                            end
                            
                            if receiver.type == SymbolTable::PrimitiveType::DOUBLE or operand.type == SymbolTable::PrimitiveType::DOUBLE
                                type = SymbolTable::PrimitiveType::DOUBLE
                            elsif receiver.type == SymbolTable::PrimitiveType::FLOAT or operand.type == SymbolTable::PrimitiveType::FLOAT
                                type = SymbolTable::PrimitiveType::FLOAT
                            elsif receiver.type == SymbolTable::PrimitiveType::INT or operand.type == SymbolTable::PrimitiveType::INT
                                type = SymbolTable::PrimitiveType::INT
                            end
                            
                            TranslationResult.new(value.to_str, type)
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                    elsif BINARY_COMPARISON_OPERATORS.include?(node.children[1])
                        operand = translateAst(node.children[2])
                        value = "(" + receiver + " " + node.children[1].to_s + " " + operand + ")"
                        
                        if receiver.type.isPrimitive and operand.type.isPrimitive
                            # implicit type conversions allowed
                            
                            if [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::BOOL)
                                if [receiver.type,  operand.type].uniq.size == 2
                                    if node.children[1] == :==
                                        # comparing objects of differnt types for identity
                                        value = "false"
                                    elsif node.children[1] == :!=
                                        value = "true"
                                    else
                                        raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL and other type"
                                    end
                                else
                                    if not [:==, :!=].include?(node.children[1])
                                        raise "ERROR: operator #{node.children[1].to_s} not applicable to BOOL"
                                    end
                                end
                            elsif [receiver.type,  operand.type].include?(SymbolTable::PrimitiveType::VOID)
                                raise "ERROR: operator #{node.children[1].to_s} not applicable to VOID"
                            end
                            
                            TranslationResult.new(value.to_str, SymbolTable::PrimitiveType::BOOL)
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                    elsif BINARY_LOGICAL_OPERATORS.include?(node.children[1])
                        operand = translateAst(node.children[2])
                        operator = node.children[1]
                        value = "(" + receiver + " " + operator.to_s + " " + operand + ")"
                        int_float = [SymbolTable::PrimitiveType::INT, SymbolTable::PrimitiveType::FLOAT]
                        type = nil
                        
                        if int_float.include?(receiver.type) and int_float.include?(operand.type)
                            if operator == :"||"
                                value = receiver
                                type = receiver.type
                            elsif operator == :"&&"
                                value = operand
                                type = operand.type
                            end
                        elsif receiver.type == SymbolTable::PrimitiveType::BOOL and operand.type == SymbolTable::PrimitiveType::BOOL
                            type = SymbolTable::PrimitiveType::BOOL
                        else
                            raise "ERROR: type inference not implemented for non-primitive types"
                        end
                        
                        TranslationResult.new(value.to_str, type)
                    else
                        if node.children[0] != nil
                            receiver += "."
                        end
                        
                        value = receiver + node.children[1].to_s + "(" + node.children[2..-1].map { |c| translateAst(c) }.join(", ") + ")"
                        # TODO: type inference for objects
                        TranslationResult.new(value, SymbolTable::PrimitiveType::INT)
                    end
                when :begin
                    if node.children.size == 1
                        translateAst(node.children[0])
                    else
                        funcId = nextFuncId
                        @functions[funcId] = node
                        funcId + "()"
                    end
                when :if
                    condition = translateAst(node.children[0])
                    
                    if (condition.type != SymbolTable::PrimitiveType::BOOL)
                        raise "ERROR: only BOOL allowed in condition but saw #{condition.type}"
                    end
                    
                    result = "if (#{condition.to_str})\n" + translateBlock(node.children[1])
                    
                    if node.children.size > 2
                        TranslationResult.new(result + "else\n" + translateBlock(node.children[2]), SymbolTable::PrimitiveType::VOID)
                    else
                        TranslationResult.new(result, SymbolTable::PrimitiveType::VOID)
                    end
                when :args
                    ""
                else
                    puts "MISSING: " + node.type.to_s
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
        
        class SymbolTable < Hash
            class PrimitiveType
                def initialize(type, cType)
                    @type = type
                    @cType = cType
                end
                
                def cTypeName
                    return @cType
                end
                
                def isPrimitive
                    return true
                end
                
                INT = self.new(:int, "int")
                FLOAT = self.new(:float, "float")
                DOUBLE = self.new(:double, "double")
                BOOL = self.new(:bool, "bool")
                VOID = self.new(:void, "void")
            end
            
            def assert(symbol, type)
                if type != self[symbol]
                    raise "ERROR: #{symbol} has type #{self[symbol]}, but requires #{type}"
                end
            end
            
            def insert(symbol, type)
                if self.has_key?(symbol)
                    raise "ERROR: #{symbol} already defined"
                end
                
                self[symbol] = type
            end
            
            def [](symbol)
                if self.has_key?(symbol)
                    return self.fetch(symbol)
                else
                    raise "ERROR: #{symbol} expected but not found in symbol table"
                end
            end
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
        puts ArrayCommand::Translator.new.translate(@block.to_source(strip_enclosure: true), "main", {@block.parameters[0][1] => ArrayCommand::Translator::SymbolTable::PrimitiveType::INT})
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
    y = x + 2 * 5.0
    y = x * 2 + 5.0
    y = y + 2
    array.map do |ppp|
    
    end
    
    foo(begin
        puts 123
        4
    end)
    
    if (y==y) == (y==y)
        puts 123
    else
        puts x
    end
    9+0
    x.foo().bar()
end
a.source