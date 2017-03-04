require "set"
require_relative "input"
require_relative "../translator/translator"
require_relative "../types/types"
require_relative "../type_aware_array"
require_relative "../parsing"
require_relative "../ast/nodes"
require_relative "../ast/lexical_variables_enumerator"
require_relative "../config/os_configuration"

Ikra::Configuration.check_software_configuration

module Ikra
    module Symbolic
        DEFAULT_BLOCK_SIZE = 256

        def self.stencil(directions:, distance:)
            return ["G", directions, distance]
        end
    
        module ParallelOperations
            def preduce(symbol = nil, **options, &block)
                if symbol == nil && (block != nil || options[:ast] != nil)
                    return ArrayReduceCommand.new(
                        to_command, 
                        block, 
                        **options)
                elsif symbol != nil && block == nil
                    ast = AST::BlockDefNode.new(
                        ruby_block: nil,
                        parameters: [:a, :b],
                        body: AST::RootNode.new(single_child:
                            AST::SendNode.new(
                                receiver: AST::LVarReadNode.new(identifier: :a),
                                selector: symbol,
                                arguments: [AST::LVarReadNode.new(identifier: :b)])))

                    return ArrayReduceCommand.new(
                        to_command,
                        nil,
                        ast: ast,
                        **options)
                else
                    raise ArgumentError.new("Either block or symbol expected")
                end
            end

            def pstencil(offsets, out_of_range_value, **options, &block)
                return ArrayStencilCommand.new(
                    to_command, 
                    offsets, 
                    out_of_range_value, 
                    block, 
                    **options)
            end

            def pmap(**options, &block)
                return pcombine(
                    **options,
                    &block)
            end

            def pcombine(*others, **options, &block)
                return ArrayCombineCommand.new(
                    to_command, 
                    wrap_in_command(*others), 
                    block, 
                    **options)
            end

            def pzip(*others, **options)
                return ArrayZipCommand.new(
                    to_command, 
                    wrap_in_command(*others),
                    **options)
            end

            def +(other)
                return pcombine(other) do |a, b|
                    a + b
                end
            end

            def -(other)
                return pcombine(other) do |a, b|
                    a - b
                end
            end

            def *(other)
                return pcombine(other) do |a, b|
                    a * b
                end
            end

            def /(other)
                return pcombine(other) do |a, b|
                    a / b
                end
            end

            def |(other)
                return pcombine(other) do |a, b|
                    a | b
                end
            end

            def &(other)
                return pcombine(other) do |a, b|
                    a & b
                end
            end

            def ^(other)
                return pcombine(other) do |a, b|
                    a ^ b
                end
            end

            def <(other)
                return pcombine(other) do |a, b|
                    a < b
                end
            end

            def <=(other)
                return pcombine(other) do |a, b|
                    a <= b
                end
            end

            def >(other)
                return pcombine(other) do |a, b|
                    a > b
                end
            end

            def >=(other)
                return pcombine(other) do |a, b|
                    a >= b
                end
            end

            # TODO(springerm): Should implement #== but this could cause trouble when using with
            # hash maps etc.

            private

            def wrap_in_command(*others)
                return others.map do |other|
                    other.to_command
                end
            end
        end

        module ArrayCommand
            include Enumerable
            include ParallelOperations

            attr_reader :block_size

            # [Fixnum] Returns a unique ID for this command. It is used during name mangling in
            # the code generator to determine the name of array identifiers (and do other stuff?).
            attr_reader :unique_id

            # An array of commands that serve as input to this command. The number of input
            # commands depends on the type of the command.
            attr_reader :input

            # Indicates if result should be kept on the GPU for further processing.
            attr_reader :keep

            # This field can only be used if keep is true
            attr_accessor :gpu_result_pointer

            # Returns the block of the parallel section or [nil] if none.
            attr_reader :block

            # A reference to the AST send node that generated this [ArrayCommand] (if inside a
            # host section).
            attr_reader :generator_node

            @@unique_id  = 1

            def self.reset_unique_id
                @@unique_id = 1
            end

            def to_s
                return "[#{self.class.to_s}, size = #{size.to_s}]"
            end

            def initialize(
                block: nil, 
                block_ast: nil, 
                block_size: nil, 
                keep: nil, 
                generator_node: nil)

                super()

                set_unique_id

                # Set instance variables
                @block_size = block_size
                @keep = keep
                @generator_node = generator_node

                if block != nil and block_ast == nil
                    @block = block
                elsif block == nil and block_ast != nil
                    @ast = block_ast
                elsif block != nil and block_ast != nil
                    raise ArgumentError.new("`block` and `block_ast` given. Expected at most one.")
                end
            end

            
            # ----- EQUALITY -----

            # Methods for equality and hash. These methods are required for comparing array
            # commands for equality. This is necessary because every array command can also
            # act as a type. Types must be comparable for equality.

            def ==(other)
                # Important: ArrayCommands may be created over and over during type inference.
                # It is important that we compare values and not identities!
                
                return self.class == other.class &&
                    block_size == other.block_size &&
                    input == other.input &&
                    keep == other.keep &&
                    block_def_node == other.block_def_node
            end

            def hash
                return (block_size.hash + input.hash + keep.hash + block_def_node.hash) % 7546777
            end

            def eql?(other)
                return self == other
            end

            # ----- EQUALITY END -----


            # ----- ARRAY METHODS -----

            def [](index)
                execute
                return @result[index]
            end

            def each(&block)
                next_index = 0

                while next_index < size
                    yield(self[next_index])
                    next_index += 1
                end
            end

            def pack(fmt)
                execute
                return @result.pack(fmt)
            end

            # ----- ARRAY END -----


            # The class or subclass of [CommandTranslator] that should be used for translating
            # this command. May be overridden.
            def command_translator_class
                return Translator::CommandTranslator
            end

            def execute
                if @result == nil
                    @result = command_translator_class.translate_command(self).execute
                end
            end
            
            def to_command
                return self
            end

            # This method is executed after execution of the parallel section has finish. The
            # boolean return value indicates if a change has been registered or not.
            def post_execute(environment)
                if keep
                    # The (temporary) result of this command should be kept on the GPU. Store a
                    # pointer to the result in global memory in an instance variable.

                    begin
                        @gpu_result_pointer = environment[("prev_" + unique_id.to_s).to_sym].to_i
                        Log.info("Kept pointer for result of command #{unique_id.to_s}: #{@gpu_result_pointer}")
                        return true  
                    rescue ArgumentError
                        # No pointer saved for this command. This can happen if the result of this
                        # command was already cached earlier and the cached result of a
                        # computation based on this command was used now. 
                        Log.info("No pointer kept for result of command #{unique_id.to_s}.")
                        return false
                    end
                end

                return false
            end

            def has_previous_result?
                return !gpu_result_pointer.nil?
            end

            # Returns a collection of the names of all block parameters.
            # @return [Array(Symbol)] list of block parameters
            def block_parameter_names
                return block_def_node.parameters
            end

            # Returns the size (number of elements) of the result, after executing the parallel 
            # section.
            # @return [Fixnum] size
            def size
                raise NotImplementedError.new
            end

            def dimensions
                # Dimensions are defined in a root command. First input currently determines the
                # dimensions (even if there are multiple root commands).
                return input.first.command.dimensions
            end

            # Returns the abstract syntax tree for a parallel section.
            def block_def_node
                if @ast == nil
                    if block == nil
                        return nil
                    end

                    # Get array of block parameter names
                    block_params = block.parameters.map do |param|
                        param[1]
                    end

                    parser_local_vars = block.binding.local_variables + block_params
                    source = Parsing.parse_block(block, parser_local_vars)
                    @ast = AST::BlockDefNode.new(
                        parameters: block_params,
                        ruby_block: block,      # necessary to get binding
                        body: AST::Builder.from_parser_ast(source))
                end

                # Ensure `return` is there
                @ast.accept(Translator::LastStatementReturnsVisitor.new)

                return @ast
            end

            # Returns a collection of lexical variables that are accessed within a parallel 
            # section.
            # @return [Hash{Symbol => Object}]
            def lexical_externals
                if block != nil
                    all_lexical_vars = block.binding.local_variables
                    lexical_vars_enumerator = AST::LexicalVariablesEnumerator.new(all_lexical_vars)
                    block_def_node.accept(lexical_vars_enumerator)
                    accessed_variables = lexical_vars_enumerator.lexical_variables

                    result = Hash.new
                    for var_name in accessed_variables
                        result[var_name] = block.binding.local_variable_get(var_name)
                    end

                    return result
                else
                    return {}
                end 
            end

            # Returns a collection of external objects that are accessed within a parallel section.
            def externals
                return lexical_externals.keys
            end

            def set_unique_id
                # Generate unique ID
                @unique_id = @@unique_id
                @@unique_id += 1
            end

            def with_index(&block)
                self.block = block
                @input.push(SingleInput.new(
                    command: ArrayIndexCommand.new(dimensions: dimensions),
                    pattern: :tid))
                return self
            end

            protected

            attr_writer :block
        end

        class ArrayIndexCommand
            include ArrayCommand
            
            attr_reader :dimensions
            attr_reader :size

            def initialize(block_size: DEFAULT_BLOCK_SIZE, keep: false, dimensions: nil)
                super(block_size: block_size, keep: keep)

                @dimensions = dimensions
                @size = dimensions.reduce(:*)

                # No input
                @input = []
            end

            def ==(other)
                return super(other) && dimensions == other.dimensions && size == other.size
            end
        end

        class ArrayCombineCommand
            include ArrayCommand

            def initialize(
                target, 
                others, 
                block, 
                ast: nil, 
                block_size: DEFAULT_BLOCK_SIZE, 
                keep: false,
                generator_node: nil,
                with_index: false)

                super(block: block, block_ast: ast, block_size: block_size, keep: keep, generator_node: generator_node)

                # Read array at position `tid`
                @input = [SingleInput.new(command: target.to_command, pattern: :tid)] + others.map do |other|
                    SingleInput.new(command: other.to_command, pattern: :tid)
                end

                if with_index
                    @input.push(SingleInput.new(
                        command: ArrayIndexCommand.new(dimensions: dimensions),
                        pattern: :tid))
                end
            end
            
            def size
                return input.first.command.size
            end

            def ==(other)
                return super(other) && size == other.size
            end
        end

        class ArrayZipCommand
            include ArrayCommand

            def initialize(target, others, **options)
                super(**options)

                @input = [SingleInput.new(command: target.to_command, pattern: :tid)] + others.map do |other|
                    SingleInput.new(command: other.to_command, pattern: :tid)
                end
            end

            def size
                return input.first.command.size
            end

            def block_parameter_names
                # Have to set block parameter names but names are never used
                return [:irrelevant] * @input.size
            end

            def ==(other)
                return super(other) && size == other.size
            end
        end

        class ArrayReduceCommand
            include ArrayCommand

            def initialize(
                target, 
                block, 
                block_size: DEFAULT_BLOCK_SIZE, 
                ast: nil,
                generator_node: nil)

                super(block: block, block_ast: ast, block_size: block_size, keep: keep, generator_node: generator_node)

                @input = [ReduceInput.new(command: target.to_command, pattern: :entire)]
            end

            def execute
                if input.first.command.size == 0
                    @result = [nil]
                elsif @input.first.command.size == 1
                    @result = [input.first.command[0]]
                else
                    @result = super
                end
            end
            
            def size
                return 1
            end

            # Returns the number of elements in the input
            def input_size
                return input.first.command.size
            end

            def ==(other)
                return super(other) && size == other.size
            end
        end

        class ArrayStencilCommand

            # This visitor executes the check_parents_index function on every local variable
            # If the local variable is the "stencil array" then its indices will get modified/corrected for the 1D array access
            class FlattenIndexNodeVisitor < AST::Visitor

                attr_reader :offsets
                attr_reader :name
                attr_reader :command

                def initialize(name, offsets, command)
                    @name = name
                    @offsets = offsets
                    @command = command
                end

                def visit_lvar_read_node(node)
                    super(node)
                    check_index(name, offsets, node)
                end

                def visit_lvar_write_node(node)
                    super(node)
                    check_index(name, offsets, node)
                end


                def check_index(name, offsets, node)
                    if node.identifier == name
                        # This is the array-variable used in the stencil

                        send_node = node
                        index_combination = []
                        is_literal = true

                        # Build the index off this access in index_combination
                        for i in 0..command.dimensions.size-1
                            send_node = send_node.parent
                            if not (send_node.is_a?(AST::SendNode) && send_node.selector == :[])
                                raise AssertionError.new(
                                    "This has to be a SendNode and Array-selector")
                            end
                            index_combination[i] = send_node.arguments.first
                            if not index_combination[i].is_a?(AST::IntLiteralNode)
                                is_literal = false
                            end
                        end

                        if is_literal
                            # The index consists of only literals so we can translate it easily by mapping the index onto the offsets

                            index_combination = index_combination.map do |x|
                                x.value
                            end

                            replacement = AST::IntLiteralNode.new(
                                value: offsets[index_combination])
                        else
                            # This handles the case where non-literals have to be translated with the Ternary Node

                            offset_arr = offsets.to_a
                            replacement = AST::IntLiteralNode.new(value: offset_arr[0][1])
                            for i in 1..offset_arr.size-1
                                # Build combination of ternary nodes

                                ternary_build = AST::SendNode.new(receiver: AST::IntLiteralNode.new(value: offset_arr[i][0][0]), selector: :==, arguments: [index_combination[0]])
                                for j in 1..index_combination.size-1

                                    next_eq = AST::SendNode.new(receiver: AST::IntLiteralNode.new(value: offset_arr[i][0][j]), selector: :==, arguments: [index_combination[j]])
                                    ternary_build = AST::SendNode.new(receiver: next_eq, selector: :"&&", arguments: [ternary_build])
                                end
                                replacement = AST::TernaryNode.new(condition: ternary_build, true_val: AST::IntLiteralNode.new(value: offset_arr[i][1]), false_val: replacement)
                            end
                        end

                        #Replace outer array access with new 1D array access

                        send_node.replace(AST::SendNode.new(receiver: node, selector: :[], arguments: [replacement]))
                    end
                end
            end

            include ArrayCommand

            attr_reader :offsets
            attr_reader :out_of_range_value
            attr_reader :use_parameter_array

            def initialize(
                target, 
                offsets, 
                out_of_range_value, 
                block, 
                ast: nil,
                block_size: DEFAULT_BLOCK_SIZE, 
                keep: false, 
                use_parameter_array: true,
                generator_node: nil,
                with_index: false)

                super(block: block, block_ast: ast, block_size: block_size, keep: keep, generator_node: generator_node)

                if offsets.first == "G"
                    # A stencil will be generated
                    dims = target.to_command.dimensions.size

                    directions = offsets[1]
                    # Directions says how many of the dimensions can be used in the stencil. E.g.: in 2D directions = 1 would relate in a stencil that only has up, down, left, right offsets, but no diagonals  
                    distance = offsets[2]
                    # Distance says how many steps can be made into the directions distance = 2 in the example before would mean 1 or 2 up/down/left/right 

                    if directions > dims
                        raise ArgumentError.new(
                            "Directions should not be higher than the number of dimensions")
                    end

                    singles = [0]
                    # Building the numbers that can be part of an offset
                    for i in 1..distance
                        singles = singles + [i] + [-i]
                    end

                    offsets = []

                    # Iterate all possibilities
                    for i in 0..singles.size**dims-1
                        # Transform current permutation to according string / base representation
                        base = i.to_s(singles.size)

                        # Fill up zeroes
                        sizedif = (singles.size**dims-1).to_s(singles.size).size - base.size
                        base = "0" * sizedif + base

                        # Check whether offset is allowed (concerning directions)
                        if base.gsub(/[^0]/, "").size >= dims - directions
                            new_offset = []
                            for j in 0..dims-1
                                new_offset.push(singles[base[j].to_i])
                            end
                            offsets.push(new_offset)
                        end
                    end
                end

                if not offsets.first.is_a?(Array)
                    offsets = offsets.map do |offset|
                        [offset]
                    end
                end

                # Read more than just one element, fall back to `:entire` for now

                @out_of_range_value = out_of_range_value
                @use_parameter_array = use_parameter_array

                if use_parameter_array
                    @input = [StencilArrayInput.new(
                        command: target.to_command,
                        pattern: :entire,
                        offsets: offsets,
                        out_of_bounds_value: out_of_range_value)]
                else
                    @input = [StencilSingleInput.new(
                        command: target.to_command,
                        pattern: :entire,
                        offsets: offsets,
                        out_of_bounds_value: out_of_range_value)]
                end

                if with_index
                    @input.push(SingleInput.new(
                        command: ArrayIndexCommand.new(dimensions: dimensions),
                        pattern: :tid))
                end
                
                # Offsets should be arrays
                for offset in offsets
                    if !offset.is_a?(Array)
                        raise ArgumentError.new("Array expected but #{offset.class} found")
                    end

                    if offset.size != dimensions.size
                        raise ArgumentError.new("#{dimensions.size} indices expected")
                    end
                end
                
                @offsets = offsets

                # Translate relative indices to 1D-indicies starting by 0
                if block_def_node != nil
                    if use_parameter_array
                        offsets_mapped = Hash.new
                        for i in 0..offsets.size-1
                            offsets_mapped[offsets[i]] = i
                        end

                        # Do not modify the original AST
                        @ast = block_def_node.clone
                        @ast.accept(FlattenIndexNodeVisitor.new(
                            block_parameter_names.first, offsets_mapped, self))
                    end
                end
            end

            def ==(other)
                return super(other) && offsets == other.offsets && 
                    out_of_range_value == other.out_of_range_value &&
                    use_parameter_array == other.use_parameter_array
            end

            def size
                return input.first.command.size
            end

            def min_offset
                return offsets.min
            end

            def max_offset
                return offsets.max
            end

            protected

            def block=(block)
                super

                if use_parameter_array
                    offsets_mapped = Hash.new
                    for i in 0..offsets.size-1
                        offsets_mapped[offsets[i]] = i
                    end

                    # Do not modify the original AST
                    @ast = block_def_node.clone
                    @ast.accept(FlattenIndexNodeVisitor.new(
                        block_parameter_names.first, offsets_mapped, self))
                end
            end
        end

        class ArraySelectCommand
            include ArrayCommand

            def initialize(target, block)
                super()

                @block = block

                # One element per thread
                @input = [SingleInput.new(command: target.to_command, pattern: :tid)]
            end
            
            # how to implement SELECT?
            # idea: two return values (actual value and boolean indicator as struct type)
        end

        class ArrayIdentityCommand
            include ArrayCommand
            
            attr_reader :target

            @@unique_id = 1

            def initialize(target, block_size: DEFAULT_BLOCK_SIZE)
                super(block_size: block_size)

                # Ensure that base array cannot be modified
                target.freeze

                # One thread per array element
                @input = [SingleInput.new(command: target, pattern: :tid)]
            end
            
            def execute
                return input.first.command
            end
            
            def size
                return input.first.command.size
            end

            def dimensions
                # 1D by definition
                return [size]
            end

            # Returns a collection of external objects that are accessed within a parallel
            # section. This includes all elements of the base array.
            def externals
                lexical_externals.keys + input.first.command
            end

            def base_type
                # TODO: add caching (`input` is frozen)
                type = Types::UnionType.new

                input.first.command.each do |element|
                    type.add(element.class.to_ikra_type)
                end

                return type
            end
        end
    end
end

class Array
    include Ikra::Symbolic::ParallelOperations

    class << self
        def pnew(size = nil, **options, &block)
            if size != nil
                dimensions = [size]
            else
                dimensions = options[:dimensions]
            end

            map_options = options.dup
            map_options.delete(:dimensions)

            return Ikra::Symbolic::ArrayIndexCommand.new(
                dimensions: dimensions, block_size: options[:block_size]).pmap(**map_options, &block)
        end
    end
    
    # Have to keep the old methods around because sometimes we want to have the original code
    alias_method :old_plus, :+
    alias_method :old_minus, :-
    alias_method :old_mul, :*
    alias_method :old_or, :|
    alias_method :old_and, :&

    def +(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            super(other)
        else
            return self.old_plus(other)
        end
    end

    def -(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            super(other)
        else
            return self.old_minus(other)
        end
    end
    
    def *(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            super(other)
        else
            return self.old_mul(other)
        end
    end

    def |(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            super(other)
        else
            return self.old_or(other)
        end
    end

    def &(other)
        if other.is_a?(Ikra::Symbolic::ArrayCommand)
            super(other)
        else
            return self.old_and(other)
        end
    end

    def to_command
        return Ikra::Symbolic::ArrayIdentityCommand.new(self)
    end
end

require_relative "host_section"
