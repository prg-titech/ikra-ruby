require "ffi"

module Demo
    extend FFI::Library
    
    ffi_lib "ffi_c"
    
    attach_function :test, [:int, :float], :int
end

module DemoKernel
    extend FFI::Library
    
    ffi_lib "kernel"
    attach_function :launch_kernel, [:float, :int, :int, :int], :int
end

puts Demo.test(1, 2.0)

puts DemoKernel.launch_kernel(1.0, 1, 1, 1)