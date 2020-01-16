#ifndef LIST
#define LIST
#include <list>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include "request.h"

using namespace std;


template <class T> 
class TSList
{
public:
TSList();
~TSList();
void pushBack(T &t);
void popFront();
unsigned int getLength();
bool isEmpty();
T* at(const int index);
T* getFront();
void eraseElement(int index);

private: 
vector<T*> mList;
mutex mMtx;
};

template <class T> TSList<T>::TSList(){
}

template <class T> TSList<T>::~TSList(){
}

template<class T> void TSList<T>::eraseElement(int index){
//check if index is in right range, return null otherwise
lock_guard<mutex> lock(mMtx);
if (index <0 || index >= mList.size()){ 
#ifdef DEBUG
fprintf(stderr, "index %d out of range for eraseElement()\n", index);
#endif
return;
}
mList.erase(mList.begin()+index);
}

template <class T> void TSList<T>::pushBack(T &t)
{
lock_guard<mutex> lock(mMtx);
mList.push_back(&t);
}

template <class T> unsigned int TSList<T>::getLength(){
lock_guard<mutex> lock(mMtx);
return mList.size();
}


template <class T> bool TSList<T>::isEmpty(){
return mList.empty();
}

template <class T> T* TSList<T>::getFront(){
lock_guard<mutex> lock(mMtx);
return  mList.front();
}
template <class T> T* TSList<T>::at(int index){
//check if index is in right range, return null otherwise
lock_guard<mutex> lock(mMtx);
if (index <0 || index >= mList.size()){ 
#ifdef DEBUG
fprintf(stderr, "index %d out of range for at()\n", index);
#endif 
return NULL;
}
return  mList.at(index);
}



#endif 
