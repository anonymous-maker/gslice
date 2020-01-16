/*
 *  Copyright (c) 2015, University of Michigan.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

/**
 * @author: Johann Hauswald, Yiping Kang
 * @contact: jahausw@umich.edu, ypkang@umich.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netdb.h>
#include <tuple>
#include <sstream>

using namespace std;

#define MAX_BUF_SIZE 65535
#define READ_MAX_BUF_SIZE 512
#define SEND_BUF_SIZE 210000

std::tuple<int, int> SOCKET_connect(int devid, bool iscpu){
    socklen_t to_len, from_len;
    int to_fd, from_fd;

    to_fd=socket(AF_UNIX, SOCK_STREAM, 0);
    from_fd=socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un to_addr;
    struct sockaddr_un from_addr;
    bzero(&to_addr, sizeof(to_addr));
    bzero(&from_addr, sizeof(from_addr));
    to_addr.sun_family=AF_UNIX;
    from_addr.sun_family=AF_UNIX;

    stringstream to_name;
    stringstream from_name;
    if(iscpu){
        to_name<<"/tmp/cpusock_input_"<<devid;
        from_name<<"/tmp/cpusock_output_"<<devid;
    }
    else{
        to_name<<"/tmp/gpusock_input_"<<devid;
        from_name<<"/tmp/gpusock_output_"<<devid;
    }
    strcpy(to_addr.sun_path, to_name.str().c_str());
    strcpy(from_addr.sun_path, from_name.str().c_str());
    to_len=sizeof(to_addr);
    from_len=sizeof(from_addr);
    if (connect(to_fd, (struct sockaddr*)&to_addr, to_len)<0){
        perror("to_fd Error");
        to_fd=-1;
    }
    if (connect(from_fd, (struct sockaddr*)&from_addr, from_len)<0){
        perror("from_fd Error");
        from_fd=-1;
    }

    return std::make_tuple(to_fd, from_fd);
}
int SOCKET_txsize(int socket, int len) {
    int ret = write(socket, (void *)&len, sizeof(int));
    //printf("ret : %d", ret);
  return ret;
}

int SOCKET_send(int socket, char *data, int size, bool debug) {
  int total = 0;
   //if (debug) printf("will send total %d bytes over socket %d \n",size,socket);

  while (total < size) {
    int sent = send(socket, data + total, size - total, MSG_NOSIGNAL);
 //   printf("returns %d \n", sent);
    if (sent <= 0) {
        printf("errno: %s \n",strerror(errno));
        break;
    }
    total += sent;
    if (debug)
      printf("Sent %d bytes of %d total via socket %d\n", total, size, socket);

    }
  return total;
}

// new wrapper for UDP socket
int UDP_SOCKET_send(int socket, char *data, int size, bool debug) {
  int total = 0;
  int offset=0;
  int sent=0;
  const unsigned int MAX_SIZE = SEND_BUF_SIZE;
    while (total < size) {
    if ( size-total >= MAX_SIZE)
    {
            sent = send(socket, data + total, MAX_SIZE, 0);
    }
    else{
            sent = send(socket, data + total, size-total, 0);
    }
   
    
    if (sent <= 0) {
            printf("errno: %s \n",strerror(errno));
            break;
    }
    total += sent;
    if (debug) printf("Sent %d bytes of %d total via socket %d\n", total, size, socket);

  }
  return total;
}


int SOCKET_rxsize(int socket) {
  int size = 0;
  int stat = read(socket, &size, sizeof(int));
  return (stat < 0) ? -1 : size;
}

int SOCKET_receive(int socket, char *data, int size, bool debug) {
  int rcvd = 0;
  while (rcvd < size) {
    int got = recv(socket, data + rcvd, size - rcvd, 0);
      if (debug)
      printf("Received %d bytes of %d total via socket %d\n", got, size,socket);
        if (got <= 0) break;
    rcvd += got;
    }
  return rcvd;
}

int CLIENT_init(char *hostname, int portno, bool debug) {
  int sockfd;
  struct sockaddr_in serv_addr;
  struct hostent *server;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    printf("ERROR opening socket\n");
    exit(0);
  }
  server = gethostbyname(hostname);
  if (server == NULL) {
    printf("ERROR, no such host\n");
    exit(0);
  }

  bzero((char *)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(portno);
  if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("ERROR connecting\n");
    return -1;
  } else {
    if (debug) printf("Connected to %s:%d\n", hostname, portno);
    return sockfd;
  }
}

int SERVER_init(int portno) {
  int sockfd, newsockfd;
  socklen_t clilen;
  struct sockaddr_in serv_addr, cli_addr;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    printf("ERROR opening socket");
    exit(0);
  }

  bzero((char *)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(portno);
  if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("ERROR on binding\n");
    exit(0);
  }
  return sockfd;
}

int SOCKET_close(int socket, bool debug) {
  if (debug) printf("Closing socket %d\n", socket);
  close(socket);
}

