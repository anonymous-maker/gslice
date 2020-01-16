#ifndef _SHARED_MEM_H__
#define _SHARED_MEM_H__ 

// Open(or Create) a shared memory segment and returns shm ID, returns 0 if failed
//int OpenSegment(const char *pathname, int seg_id);

//cretes(or open) shmem segment, returns shmem id, called by server
int SHMEM_create(const char *pathname, int proj_id);

// attach shmem to provided pointers, 
int SHEMEM_register(int shmem_id, void* pData);

int SHMEM_putInput(int jobID, int size ,void* vp, float* input);
int SHMEM_getInput(void* pData, float* input, int *jobid);

int SHMEM_putOutput( void *pData, int *output);
int SHMEM_getOuput(int shmem_id, int *pData, int *ret);

int SHMEM_deregister(void* pData);

// in
int SHMEM_destroy(int shmemid);

#else
#endif
