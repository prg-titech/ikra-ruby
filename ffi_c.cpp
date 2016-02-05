//#include "ffi_c.h"
#include <stdlib.h>

extern "C"  __declspec(dllexport)
int *test(int a, float b)
{
	int *x = (int*) malloc(sizeof(int) * 4);
    x[0] =97;
    x[1] =98;
    x[2] = 99;
    x[3]=100;
    
    return x;
}

