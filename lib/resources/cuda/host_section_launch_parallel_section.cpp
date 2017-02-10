({
    /*{array_command_type}*/ cmd = /*{array_command}*/;

    if (cmd->result == 0) {
        /*{kernel_invocation}*/
        cmd->result = /*{kernel_result}*/;
    }

    cmd;
})