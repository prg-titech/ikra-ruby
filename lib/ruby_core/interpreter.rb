module Ikra
    module RubyIntegration
        # No need to do type inference or code generation, if a method is called on an
        # on an instance of one of these classes.
        INTERPRETER_ONLY_CLS_OBJ = [Ikra::Symbolic.singleton_class]


        def self.is_interpreter_only?(type)
            if !type.is_a?(Types::ClassType)
                return false
            end

            return INTERPRETER_ONLY_CLS_OBJ.include?(type.cls)
        end
    end
end