require 'rake/testtask'

Rake::TestTask.new("test:unit") do |t|
    t.test_files = FileList.new('test/unit/*.rb') do |fl|
        # The following tests are broken
        # fl.exclude("test/unit/host_section_test.rb")
    end

    t.warning = false
end

Rake::TestTask.new("test:codegen") do |t|
    t.test_files = FileList['test/test_codegen.rb']
    t.warning = false
end

Rake::TestTask.new("test:benchmark") do |t|
    file_list = FileList['test/benchmarks/*.rb']
    file_list.exclude("test/benchmarks/benchmark_base.rb")

    t.test_files = file_list
    t.warning = false
end

task :"test:unit" do
    Rake::Task["test:unit"].invoke
end

task :"test:codegen" do
    Rake::Task["test:codegen"].invoke
end

task :"test:benchmark" do
    Rake::Task["test:benchmark"].invoke
end

task :"test" do
    Rake::Task["test:unit"].invoke
    Rake::Task["test:codegen"].invoke
    Rake::Task["test:benchmark"].invoke
end

desc "Run tests"
task :default => :test
