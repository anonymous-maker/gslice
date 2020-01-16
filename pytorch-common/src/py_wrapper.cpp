#include <vector>
#include "py_wrapper.h"


int initPy(const char *new_path){
    if (new_path != NULL)
        setenv("PYTHONPATH", new_path, 1);
    else
         setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
    return 0;
}

int exitPy(){
     Py_Finalize();
     return 0;

}




PyObject*  callPyFunc(const char* pyname, const char* funcname, PyObject *pArgs){
    PyObject *pName, *pModule, *pFunc;
    PyObject *pValue;
    pValue=NULL;
    pName = PyUnicode_DecodeFSDefault(pyname);
    /* Error checking of pName left out */
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, funcname);
        /* pFunc is a new reference */
         if (pFunc && PyCallable_Check(pFunc)) {            
            assert(pArgs != NULL);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                //printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                //output = pValue;
                //printf("%ld \n", Py_REFCNT(pValue));

            //  Py_DECREF(pValue);
            }
            else{
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return NULL;
            }
        }
        else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", funcname);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
     }
     else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", pyname);
        return NULL;
     }
    return pValue;
}
PyObject* formArgument(int argc, void** vargs, FUNC_PARAMS type){ // will need tp come up with a better way to do this... maybe enum?
    PyObject *pArgs, *pValue;
    pArgs = PyTuple_New(argc);
    switch(type){
            case DEFAULT:
            {
                    char **argv1 = (char **)vargs;
                    for (int i = 0; i < argc; ++i) {
                            //printf("%s \n", argv[i+2]);
                            pValue = PyUnicode_FromString(argv1[i]);
                            if (!pValue) {
                                    Py_DECREF(pArgs);
                                    fprintf(stderr, "Cannot convert argument\n");
                                    return NULL;
                            }
                            /* pValue reference stolen here: */
                            PyTuple_SetItem(pArgs, i, pValue);
                    }
                    break;   
            } // case DEFAULT
            case METHOD:
            {
                    void **argv2=vargs;
                    for (int i = 0; i < argc; ++i) {
                            pValue=(PyObject *)argv2[i+2];
                            if (!pValue) {
                                    Py_DECREF(pArgs);
                                    fprintf(stderr, "Cannot convert argument\n");
                                    return NULL;
                            }
                            /* pValue reference stolen here: */
                            PyTuple_SetItem(pArgs, i, pValue);
                    }
                    break;
            } // case METHOD
            case POS_PREPROCESS:
            {
                    void **argv3=vargs;
                    /*batch vector of char* into list of string for python*/
                    std::vector<char*> *sents = (std::vector<char*> *)argv3[0];
                    int vsize = sents->size();
                    PyObject *plist =  PyList_New(vsize);
                    assert(plist != NULL);
                    for(int i =0;i<sents->size(); i++){
                        pValue = PyUnicode_FromString((sents->at(i)));
                        if (!pValue) {
                            Py_DECREF(pArgs);
                            fprintf(stderr, "Cannot convert argument\n");
                            return NULL;
                        }
                        PyList_SetItem(plist, i, pValue);
                    }
                    PyTuple_SetItem(pArgs, 0, plist);
                    /*insert dict into arg tuple*/
                    PyObject *dict = (PyObject *)argv3[1];
                    if (!dict) {
                            Py_DECREF(pArgs);
                            fprintf(stderr, "Cannot convert argument\n");
                            return NULL;
                    }
                    PyTuple_SetItem(pArgs, 1, dict);
                    /*insert max_len into arg tuple*/
                    int *max_len = (int *)argv3[2];
                    PyObject *pmax = PyLong_FromLong(*max_len);
                    if (!pmax) {
                            Py_DECREF(pArgs);
                            fprintf(stderr, "Cannot convert argument\n");
                            return NULL;
                    }
                    PyTuple_SetItem(pArgs, 2, pmax);
                    break;
            } //case POS_PREPROCESS
            case CHAT_PREPROCESS:
            {
                    void **argv3=vargs;
                    /*batch vector of char* into list of string for python*/
                    char* sent =(char *)argv3[0];
                    pValue = PyUnicode_FromString(sent);
                    if (!pValue) {
                        Py_DECREF(pArgs);
                        fprintf(stderr, "Cannot convert argument\n");
                        return NULL;
                    }                                        
                    PyTuple_SetItem(pArgs, 0, pValue);
                    /*insert dict into arg tuple*/
                    PyObject *dict = (PyObject *)argv3[1];
                    if (!dict) {
                            Py_DECREF(pArgs);
                            fprintf(stderr, "Cannot convert argument\n");
                            return NULL;
                    }
                    PyTuple_SetItem(pArgs, 1, dict);
                    /*insert max_len into arg tuple*/
                    int *max_len = (int *)argv3[2];
                    PyObject *pmax = PyLong_FromLong(*max_len);
                    if (!pmax) {
                            Py_DECREF(pArgs);
                            fprintf(stderr, "Cannot convert argument\n");
                            return NULL;
                    }
                    PyTuple_SetItem(pArgs, 2, pmax);
                    break;
            } //case CHAT_PREPROCESS
            default:
                    fprintf(stderr, "unrecognized argument!\n");
                     return NULL;
    }// switch(type)
        return pArgs;
}                                                                                                                             



