// header file for storing all states useds throughout the server
#ifndef _STATE_H__
#define _STATE_H__


namespace djinn{
enum TaskState {EMPTY=0,QUEUED=1,FULL=2,BATCHED=3,RUNNING=4,END=5}; //BATCHED  is only used for 'requests'(not batched request), 
enum Backend { caffe=0, pytorch=1 };
enum TensorDataOption {KFLOAT32=0,KINT64=1};
}
#else
#endif

