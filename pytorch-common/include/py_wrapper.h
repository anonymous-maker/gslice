#ifndef _PY_WRAPPER_H__
#define _PY_WRAPPER_H__
#include <Python.h>
#include <map> 
enum FUNC_PARAMS {DEFAULT, METHOD, POS_PREPROCESS, CHAT_PREPROCESS};

PyObject*  callPyFunc(const char* pyname, const char* funcname, PyObject *pArgs);
PyObject* formArgument(int argc, void** vargs, FUNC_PARAMS type);                                                                 
int initPy(const char *new_path);
int exitPy();
#else
#endif
