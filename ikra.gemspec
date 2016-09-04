require 'rake'

Gem::Specification.new do |s|
  s.name        = 'ikra'
  s.version     = '0.0.1'
  s.date        = '2016-08-04'
  s.summary     = "Ikra"
  s.description = "GPGPU Accelerator for Array-based Computations"
  s.authors     = ["Matthias Springer"]
  s.email       = 'matthias.springer@acm.org'
  s.files       = FileList["lib/**/*"].to_a
  s.homepage    = 'https://github.com/prg-titech/ikra-ruby'
  s.license     = 'MIT'

  s.add_runtime_dependency 'ffi', '~> 1.9.14', '>= 1.9.14'
  s.add_runtime_dependency 'parser', '~> 2.3.1.2', '>= 2.3.1.2'
  s.add_runtime_dependency 'method_source', '~> 0.8.2', '>= 0.8.2'

  s.required_ruby_version = '>= 2.3.0'
end