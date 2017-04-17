require 'rake'

Gem::Specification.new do |s|
  s.name        = 'ikra'
  s.version     = '0.0.2'
  s.date        = '2017-04-17'
  s.summary     = "Ikra"
  s.description = "GPGPU Accelerator for Array-based Computations"
  s.authors     = ["Matthias Springer"]
  s.email       = 'matthias.springer@acm.org'
  s.files       = FileList["lib/**/*"].to_a
  s.homepage    = 'https://prg-titech.github.io/ikra-ruby/'
  s.license     = 'MIT'

  s.add_dependency 'ffi', '~> 1.9.14', '>= 1.9.14'
  s.add_dependency 'parser', '~> 2.3.1.2', '>= 2.3.1.2'
  s.add_dependency 'method_source', '~> 0.8.2', '>= 0.8.2'

  # Sourcify dependencies
  s.add_dependency 'ruby2ruby', '~> 1.3.1'
  s.add_dependency 'sexp_processor', '~> 3.2.0'
  s.add_dependency 'ruby_parser', '~> 2.3.1'
  s.add_dependency 'file-tail', '~> 1.0.10'

  s.add_development_dependency 'bacon'
  s.add_development_dependency 'pry'

  s.required_ruby_version = '>= 2.3.0'
end