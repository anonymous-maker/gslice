#include "py_wrapper.h"
#include "nlp_utils.h"

PyObject* initDict(const char* pyname, const char* funcname, const char* path_to_tsv){
    const char *arguments[1];
    arguments[0]=path_to_tsv;
    PyObject *pargs = formArgument(1,(void **)arguments,DEFAULT);
    return callPyFunc(pyname, funcname, pargs);
   
}
PyObject* preprocessData_POS(const char* pyname, const char* funcname, std::vector<char*> *sents, PyObject *dict, int max_len){
    void  *arguments[3];
    arguments[0]=(void*)sents;
    arguments[1]=(void*)dict;
    arguments[2]=(void*)&max_len;
    PyObject *pargs =  formArgument(3,(void **)arguments,POS_PREPROCESS);
    return callPyFunc(pyname,funcname ,pargs);
}
PyObject* preprocessData_CHATBOT(const char* pyname, const char* funcname, const char* sent, PyObject *dict, int max_len){
   void  *arguments[3];
    arguments[0]=(void*)sent;
    arguments[1]=(void*)dict;
    arguments[2]=(void*)&max_len;        
    PyObject *pargs =  formArgument(3,(void **)arguments,CHAT_PREPROCESS);
    return callPyFunc(pyname,funcname ,pargs);

}

