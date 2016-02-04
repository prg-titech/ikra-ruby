//#include "ffi_c.h"

extern "C"  __declspec(dllexport)
int test(int a, float b)
{
	return a + b;
}

