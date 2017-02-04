({
    /*{array_command_type}*/ cmd = /*{array_command}*/;

    if (cmd->result == 0) {
        /*{kernel_invocation}*/
        cmd->result = /*{kernel_result}*/;
    }

    fixed_size_array_t</*{result_inner_type}*/>(cmd->result, /*{result_size}*/);
})