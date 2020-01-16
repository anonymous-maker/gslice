#include <torch/script.h> // One-stop header.
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <memory>
#include <sys/time.h>
#include <pthread.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "socket.h"
#include "tonic.h"
#include "cvmat_serialization.h"
#include "img_utils.h"
#include "torchutils.h"
#include "cpuUsage.h"
#include "common_utils.h" //printTimeStamp moved to here
#include "shared_mem.h"

#include <sys/socket.h>
#include <sys/un.h>

#define SOCK_FLAG 0
using namespace std;

vector<string> netlist={"vgg16.pt", "resnet18.pt", "alexnet.pt", "squeezenet.pt", "dcgan-gpu.pt"};
std::unordered_map<std::string, int> mapping={{"vgg16.pt", 0}, {"resnet18.pt", 1}, {"alexnet.pt", 2}, {"squeezenet.pt", 3}, {"dcgan-gpu.pt", 4}};
vector<string> netlist_cpu={"vgg16.pt", "resnet18.pt", "alexnet.pt", "squeezenet.pt", "dcgan-cpu.pt"};
std::unordered_map<std::string, int> mapping_cpu={{"vgg16.pt", 0}, {"resnet18.pt", 1}, {"alexnet.pt", 2}, {"squeezenet.pt", 3}, {"dcgan-cpu.pt", 4}};

namespace po=boost::program_options;

mutex iqueue_mtx;
condition_variable iqueue_cv;
mutex oqueue_mtx;
condition_variable oqueue_cv;

struct queue_elem{
    int reqid;
    int jobid;
    std::vector<int64_t> dims;
    float* indata;
};

queue<struct queue_elem*> input_queue;
queue<struct queue_elem*> output_queue;


int devid;
int threadcap;
int dedup;
int use_cpu;
bool CLIENT_DISCONNECT=false;
std::string common_dir;

const char *SEND="SEND";
const char *RECV="RECV";
const char *COMP="COMP";
const char *LISTENING="LISTENING";
const char *ACCEPTED="ACCEPTED";
const char *DONE="INIT DONE";
const char *START="INIT START";


bool exitFlag = false;

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
      ("common,com", po::value<std::string>()->default_value("../../pytorch-common/"),
       "Directory with configs and weights")
      ("devid,d", po::value<int>()->default_value(-1),"Device ID")
    ("threadcap,tc", po::value<int>()->default_value(100),"thread cap(used for designation)")
    ("dedup,dn", po::value<int>()->default_value(0),"identifier between same device and cap")


      ("cpu,c", po::value<int>()->default_value(0),"Use CPU");
  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}
static __inline__ unsigned long long rdtsc(void)
{
        unsigned long long int x;
        __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
        return x;
}

void* recv_input(void* vp){

  printTimeStampWithName(RECV, START);
    int server_sock, client_sock, rc;
    socklen_t len;
    int i;
    int bytes_rec = 0;
    struct sockaddr_un server_sockaddr;
    struct sockaddr_un client_sockaddr;
    int cur_read;
    int backlog = 10;
    struct queue_elem* q;
            
    memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
    memset(&client_sockaddr, 0, sizeof(struct sockaddr_un));
            
    stringstream sockname;
    if(use_cpu){
        sockname<<"/tmp/cpusock_input_"<<devid<<"_"<<threadcap<<"_"<<dedup;
    }           
    else{   
        sockname<<"/tmp/gpusock_input_"<<devid<<"_"<<threadcap<<"_"<<dedup;
    }       
                
    server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR: %s\n", strerror(errno));
        exit(1);
    }   
    server_sockaddr.sun_family = AF_UNIX;
    strcpy(server_sockaddr.sun_path, sockname.str().c_str());
    len=sizeof(server_sockaddr);
        
    unlink(sockname.str().c_str());
    rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
    if (rc == -1){
        printf("BIND ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    rc = listen(server_sock, backlog);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }
    printTimeStampWithName(server_sockaddr.sun_path, LISTENING);
    while (1){
    client_sock = accept(server_sock, (struct sockaddr *) &client_sockaddr, &len);
    if (client_sock == -1){
        printf("ACCEPT ERROR: %s\n", strerror(errno));
        close(server_sock);
        close(client_sock);
        exit(1);
    }
//#ifdef DEBUG
    printTimeStampWithName( server_sockaddr.sun_path, ACCEPTED);
//#endif 
    
    while(1){
        int ret;
        int dimlen=0;
        int buf=0;
        int datalen=0;
        if (ret=read(client_sock,&dimlen, sizeof(int)) <=0){
           // printf("client Disconnected  \n");
            break;
        }
        uint64_t start = getCurNs();
        assert(dimlen <=4);
        if(dimlen!=0){
            q=new queue_elem();
            for(int i =0; i <dimlen; i++){
            if ((ret=read(client_sock,&buf,sizeof(int))) > 0){
                    q->dims.push_back(buf);
                }
            }
            if ((ret=read(client_sock,&datalen,sizeof(int))) > 0){
            }
            q->indata=(float*)malloc(datalen*sizeof(float));
#ifdef DEBUG
            printf("[%d][%d]receive %u bytes \n",devid ,threadcap,datalen*sizeof(float));
        uint64_t start2 = getCurNs();
#endif 
            if (ret=SOCKET_receive(client_sock, (char*)q->indata, datalen*sizeof(float), false) <=0){
                printf("ERROR in receiving input data\n ");
            }
#ifdef DEBUG
        printf("----------- tsc: %llu\n", rdtsc());
        uint64_t end2 = getCurNs();
#endif 
            buf=0;
            if (ret=read(client_sock, &buf, sizeof(int)) >0){
                q->reqid=buf;
            }
            buf=0;
            if (ret=read(client_sock, &buf, sizeof(int)) >0){
                q->jobid=buf;
            }
            input_queue.push(q);
#ifdef DEBUG
        uint64_t end = getCurNs();
            printf("input data recv latency %lu \n",(end2-start2)/1000000);
            printf("total recv_latency: %lu \n", (end-start)/1000000);
#endif 
            iqueue_cv.notify_one();           
        }
            else{
                printf("read returned 0. stop reading \n");
                break;
            }
            }// inner loop
        SOCKET_close(client_sock,true);
         CLIENT_DISCONNECT=true;
       oqueue_cv.notify_one();

        }//outer loop
    
}

pthread_t init_input_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, recv_input, NULL)!=0){
        printf("init_input_thread: Error\n");
    }
    return tid;
}



void* send_output(void* vp){

      printTimeStampWithName(SEND, START);
    int server_sock, client_sock, rc;
    socklen_t len;
    int i;
    int bytes_rec = 0;
    struct sockaddr_un server_sockaddr; 
    struct sockaddr_un client_sockaddr;
    int cur_read;
    int backlog = 10;
    struct queue_elem* q;
            
    memset(&server_sockaddr, 0, sizeof(struct sockaddr_un));
    memset(&client_sockaddr, 0, sizeof(struct sockaddr_un));
            
    stringstream sockname;
    if(use_cpu){
        sockname<<"/tmp/cpusock_output_"<<devid<<"_"<<threadcap<<"_"<<dedup;
    }           
    else{   
        sockname<<"/tmp/gpusock_output_"<<devid<<"_"<<threadcap<<"_"<<dedup;
    }       
                
    server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock == -1){
        printf("SOCKET ERROR: %s\n", strerror(errno));
        exit(1);
    }   
    server_sockaddr.sun_family = AF_UNIX;
    strcpy(server_sockaddr.sun_path, sockname.str().c_str());
    len=sizeof(server_sockaddr);
        
    unlink(sockname.str().c_str());
    rc = bind(server_sock, (struct sockaddr *) &server_sockaddr, len);
    if (rc == -1){
        printf("BIND ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }

    rc = listen(server_sock, backlog);
    if (rc == -1){ 
        printf("LISTEN ERROR: %s\n", strerror(errno));
        close(server_sock);
        exit(1);
    }
    printTimeStampWithName(server_sockaddr.sun_path, LISTENING);
    while(1){
            client_sock = accept(server_sock, (struct sockaddr *) &client_sockaddr, &len);
            if (client_sock == -1){
                printf("ACCEPT ERROR: %s\n", strerror(errno));
                close(server_sock);
                close(client_sock);
                exit(1);
            }
        //#ifdef DEBUG
            printTimeStampWithName(server_sockaddr.sun_path, ACCEPTED);
        //#endif 
            while(1){
                unique_lock<mutex> lock(oqueue_mtx);
                oqueue_cv.wait(lock, []{return !output_queue.empty() || CLIENT_DISCONNECT;});
                if (CLIENT_DISCONNECT){
                    CLIENT_DISCONNECT=false;
                    break;
                }
                q=output_queue.front();
                output_queue.pop();

                //SOCKET_txsize(client_sock, q->reqid);
                int rid = q->reqid;
                write(client_sock, (char*)&rid, sizeof(int));
#ifdef DEBUG
                printf("send output rid : %d \n",rid);
#endif 
                free(q);
            }
            SOCKET_close(client_sock, true);
    }

}
pthread_t init_output_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, send_output, NULL)!=0){
        printf("init_output_thread: Error\n");
    }
    return tid;
}

void* compute(void* vp){
    printTimeStampWithName(COMP, START);
    struct queue_elem* q;
    torch::Device cpu_dev(torch::kCPU);
    torch::Device gpu_dev(torch::kCUDA,0);
    std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> net_map;
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input;
    float* buffer;
    if(use_cpu){
        for(auto i: netlist_cpu){
            net_map[mapping[i]]=torch::jit::load((common_dir+i).c_str());
            if(i=="dcgan-cpu.pt"){
                std::vector<int64_t> dims={1,100,1,1};
                buffer = (float *)malloc(1*100*sizeof(float));
                memset(buffer, 0, 1*100*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            else{
                std::vector<int64_t> dims={1,3,152,152};
                buffer = (float *)malloc(1*3*152*152*sizeof(float));
                memset(buffer, 0, 1*3*152*152*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            inputs.push_back(input);
            net_map[mapping[i]]->forward(inputs).toTensor();
            inputs.clear();
            free(buffer);
        }
    }
    else{
        for(auto i: netlist){
            //cout<<i<<endl;
            net_map[mapping[i]]=torch::jit::load((common_dir+i).c_str());
            net_map[mapping[i]]->to(gpu_dev);
            if(i=="dcgan-gpu.pt"){
                std::vector<int64_t> dims={1,100,1,1};
                buffer = (float *)malloc(1*100*sizeof(float));
                memset(buffer, 0, 1*100*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            else{
                std::vector<int64_t> dims={1,3,152,152};
                buffer = (float *)malloc(1*3*152*152*sizeof(float));
                memset(buffer, 0, 1*3*152*152*sizeof(float));
                torch::TensorOptions options(torch::kFloat32);
                input =  torch::from_blob(buffer, torch::IntList(dims), options);
            }
            input=input.to(gpu_dev);
            inputs.push_back(input);
            net_map[mapping[i]]->forward(inputs).toTensor();
            cudaDeviceSynchronize();
            inputs.clear();
            free(buffer);
        }
    }
    printTimeStampWithName(COMP, DONE);
    bool is_empty;
    torch::TensorOptions options(torch::kFloat32);
    torch::Tensor t;
    torch::Tensor out;
    while(1){
        //compute here
        unique_lock<mutex> lock(iqueue_mtx);
        iqueue_cv.wait(lock, []{return !input_queue.empty();});
        q=input_queue.front();
        input_queue.pop();
        
        t=convert_rawdata_to_tensor(q->indata, q->dims, options);
        if(!use_cpu){
            t=t.to(gpu_dev);
        }
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(t);
        out=net_map[q->jobid]->forward(inputs).toTensor();
        if(!use_cpu){
            cudaDeviceSynchronize();
        }
        //send output
        output_queue.push(q);
        oqueue_cv.notify_one();
    }
}

pthread_t init_compute_thread(){
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8*1024 * 1024);
    pthread_t tid;

    if(pthread_create(&tid, &attr, compute, NULL)!=0){
        printf("init_compute_thread: Error\n");
    }
    return tid;
}
int main(int argc, char** argv){
    pthread_t compute, send, recv;
    po::variables_map vm=parse_opts(argc, argv);
    devid=vm["devid"].as<int>();
    dedup=vm["dedup"].as<int>();
    //int shmem_id = SHMEM_create("test_shmem", devid+1);
    //void* shared_memory;
    //SHEMEM_register(shmem_id,shared_memory);
    //SHMEM_deregister(shared_memory);
    //SHMEM_destroy(shmem_id);
    threadcap=vm["threadcap"].as<int>();
    common_dir=vm["common"].as<string>() + "models/";
    use_cpu=vm["cpu"].as<int>();
    stringstream ss;
    ss<<"/tmp/nvidia-mps";
    if(!use_cpu){
        if(devid<4){
            setenv("CUDA_MPS_PIPE_DIRECTORY", ss.str().c_str(), 1);
            }
    }
    compute=init_compute_thread();
    recv=init_output_thread();
    send=init_input_thread();
    pthread_join(compute, NULL);
    pthread_join(send, NULL);
    pthread_join(recv, NULL);

    return 0;
}
