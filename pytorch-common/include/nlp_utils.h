#ifndef _NLP_UTIL_H__
#define _NLP_UTIL_H__
#include <Python.h>
#include <vector>

PyObject* initDict(const char* pyname, const char* funcname, const char* path_to_tsv);

PyObject* preprocessData_POS(const char* pyname, const char* funcname, std::vector<char*> *sents, PyObject *dict, int max_len);
PyObject* preprocessData_CHATBOT(const char* pyname, const char* funcname, const char* sent, PyObject *dict, int max_len);

#else
#endif
