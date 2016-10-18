require 'rake/testtask'

Rake::TestTask.new("test:unit") do |t|
    t.test_files = FileList['test/unit/*.rb']
    t.warning = false
end

Rake::TestTask.new("test:codegen") do |t|
    t.test_files = FileList['test/test_codegen.rb']
    t.warning = false
end

task :test do
    Rake::Task["test:unit"].invoke
    Rake::Task["test:codegen"].invoke
end

desc "Run tests"
task :default => :test
