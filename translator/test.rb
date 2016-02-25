require_relative "block_translator.rb"
require_relative "translator.rb"
require_relative "../types/primitive_type"
require_relative "../scope"

magnify = 1.0
hx_res = 2000
hy_res = 2000
iter_max = 256

block = Proc.new do |j|
    hx = j % hx_res
    hy = j / hx_res
    
    cx = (hx.to_f / hx_res.to_f - 0.5) / magnify*3.0 - 0.7
    cy = (hy.to_f / hy_res.to_f - 0.5) / magnify*3.0
    
    x = 0.0
    y = 0.0
    
    for iter in 0..100
        xx = x*x - y*y + cx
        y = 2.0*x*y + cy
        x = xx
        
        if x*x + y*y > 100
            break
        end
    end
    
    iter % 256
end

scope = Scope.new
puts Ikra::Translator.translate_block(block: block, 
    symbol_table: scope, 
    env_builder: Ikra::Translator::EnvironmentBuilder.new, 
    input_types: [[PrimitiveType::Int].to_set])
