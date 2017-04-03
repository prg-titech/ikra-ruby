# Ikra
Ikra is an array-based parallel extension to Ruby with dynamic compilation. The high-level goal of the Ikra project is to allow developers to exploit GPU-based high-performance computing without paying much attention to details of the underlying GPU infrastructure and CUDA. This is the Ruby version of Ikra  (Ikra implemented in Ruby).

## Installation
* Tested with Ruby 2.3.0
* Run `bundle install`
* clone sourcify lib
* run `cd lib ; git clone https://github.com/matthias-springer/sourcify.git`
* run `cd sourcify; bundle instal`
* Make sure that the nVidia CUDA Toolkit is installed (check paths in `config/os_configuration.rb`)
