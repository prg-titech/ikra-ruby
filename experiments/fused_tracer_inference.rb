# Object Tracer and Type Inference Engine
# Constraints:
# 1. Process only read/written instance variables
# 2. Polymorphic types are allowed
# Consequences:
# 1. Reading instance variable --> new part of object graph becomes visible --> new type information for already traced objects --> need to rerun type inference for methods accessing inst var 

objs = base + lexical				# type -> [obj]
methods = {(nil, nil) => block}		# type x name -> MethodDefinition
inst_vars = {}						# type x name -> InstVarDefinition
worklist = []

def type_inference_and_tracer(block)
	type_inference(block)

	while (!worklist.empty()) {
		next = worklist.pop()
		if (type_inference(next)) {
			worklist.add_all(next.callers)
		}
	}
end

def type_inference(method)
	case node.type
		when :method_call
			if not methods.has_key?(receiver.type, selector)
				callee.parameters.type = arguments.type
				type_inference(callee)
			elsif not callee.parameters.include?(arguments.type)
				callee.parameters.type.extend(arguments.type)
				if type_inference(callee)
					worklist.add_all(callee.callers - [method])
				end
			else
				# reuse return value
			end

			callee.callers.add(method)
			type = callee.return_type
		when :inst_var_read
			if not inst_vars.has_key?(class_name, inst_var_name)
				obj_trace(class_name, inst_var_name)
			end

			var.add_reader(method)
			type = var.type
		when :inst_var_write
			if not inst_vars.has_key?(class_name, inst_var_name)
				obj_trace(class_name, inst_var_name)
			end

			if not var.type.includes?(value.type)
				var.type.extend(value.type)
				worklist.add_all(var.readers - [method])
			end

			type = value.type
	end
end

def check_var_type(var, type)
	if not var.type.include?(type)
		var.type.extend(type)
		worklist.add_all(var.readers)
	end
end

def obj_trace(class_name, inst_var_name)
	for obj in objs[class_name]
		value = obj.instance_variable_get(inst_var_name)
		var = inst_vars[(class_name. inst_var_name)]
		check_var_type(var, value.type)
		single_obj_trace(value)
	end
end

def single_obj_trace(obj)
	if not objs[obj.type].include?(obj)
		# not yet traced

		for inst_var_name in inst_vars[obj.type]
			value = obj.instance_variable_get(inst_var_name)
			check_var_type(inst_vars[(obj.type, inst_var_name)], value)
			single_obj_trace(value)
		end
	end
end

