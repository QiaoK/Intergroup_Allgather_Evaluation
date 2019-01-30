#include "mpi.h"
#include <unistd.h> /* getopt() */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "random.h"
#define FLOATING_CHECK(a,b) (a!=b)
#define DEBUG 0
#define ERR { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: (%s)\n", __LINE__,errorString); \
    } \
}
#define MAP_DATA(a,b) ((a)*123+(b)*653+14*((a)-742)*((b)-15))
int err;

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [OPTION]... [FILE]...\n"
    "       [-h] Print help\n"
    "       [-s] number of MPI processes in group A\n"
    "       [-r] number of MPI processes in group B\n"
    "       [-a] message count per process in group A\n"
    "       [-b] message count per process in group B\n"
    "       [-d] message block unit size\n"
    "       [-n] number of trials for the experiments\n"
    "       [-p] only significant for Allgatherv (when t = 3, 4)\n"
    "           0: regular message distribution\n"
    "           1: irregular message distribution\n"
    "       [-t] which evaluation\n"
    "           0: Intergroup Allgather with native MPI library\n"
    "           1: Intergroup Allgather with algorithm 2 of EuroMPI paper\n"
    "           2: Intergroup Allgather with algorithm 1 of PARCO paper\n"
    "           3: Intergroup Allgatherv with native MPI library\n"
    "           4: Intergroup Allgather with algorithm 2 of PARCO paper\n";
    fprintf(stderr, help, argv0);
}



/*
 * Initialize send buffer for intergroup Allgather.
*/
void create_data(int rank, int n_senders,int n_receivers,int **send_buff,int** receive_buff,int *sendcount,int *recvcount,int dim_x,int size1,int size2){
	int i;
	if(rank>=n_senders){
	        *sendcount = dim_x*size2;
	        *recvcount = dim_x*size1;
	        receive_buff[0]=(int*)malloc(recvcount[0]*n_senders*sizeof(int));
	}else{
	        *sendcount = dim_x*size1;
	        *recvcount = dim_x*size2;
	        receive_buff[0]=(int*)malloc(recvcount[0]*n_receivers*sizeof(int));
	}
	send_buff[0]=(int*)malloc(sendcount[0]*sizeof(int));
        for(i=0;i<sendcount[0];i++){
            send_buff[0][i]=MAP_DATA(rank,i);
        }
}
/*
 * Initialize send buffer for intergroup Allgatherv.
*/
void create_vector_data(int rank, int n_senders,int n_receivers,int **send_buff,int** receive_buff,int *sendcount,int *recvcount,int dim_x){
	int i,size1=0,size2=0,total_send_messages;
	for(i=0;i<n_senders;i++){
		sendcount[i]*=dim_x;
		size1+=sendcount[i];
	}
	for(i=0;i<n_receivers;i++){
		recvcount[i]*=dim_x;
		size2+=recvcount[i];
	}
	if(rank>=n_senders){
		total_send_messages=recvcount[rank-n_senders];
	        receive_buff[0]=(int*)malloc(size1*sizeof(int));
		send_buff[0]=(int*)malloc(total_send_messages*sizeof(int));
	}else{
		total_send_messages=sendcount[rank];
	        receive_buff[0]=(int*)malloc(size2*sizeof(int));
	        send_buff[0]=(int*)malloc(total_send_messages*sizeof(int));
	}
        for(i=0;i<total_send_messages;i++){
            send_buff[0][i]=MAP_DATA(rank,i);
        }
}
/*
 * Check if this process has received correct messages at the end of intergroup Allgatherv.
*/
void validate_vector_result(int rank,int n_senders,int n_receivers,int *receive_buff,int *recvcounts){
	int i,j;
	int size=0;
	if(rank<n_senders){
		for(i=0;i<n_receivers;i++){
			for(j=0;j<recvcounts[i];j++){
				if(receive_buff[size+j]!=MAP_DATA(i+n_senders,j)){
					printf("rank %d: test failed at i=%d,j=%d,%d!=%d\n",rank,i,j,receive_buff[size+j],MAP_DATA(i+n_senders,j));
				}
			}
			size+=recvcounts[i];
		}
	}else{
		for(i=0;i<n_senders;i++){
			for(j=0;j<recvcounts[i];j++){
				if(receive_buff[size+j]!=MAP_DATA(i,j)){
					printf("rank %d: test failed at i=%d,j=%d,%d!=%d\n",rank,i,j,receive_buff[size+j],MAP_DATA(i,j));
				}
			}
			size+=recvcounts[i];
		}
	}
}
/*
 * Check if this process has received correct messages at the end of intergroup Allgather.
*/
void validate_result(int rank,int n_senders,int n_receivers,int *receive_buff, int recvcount){
	int i,j;
        if(rank<n_senders){
		for(i=n_senders;i<n_senders+n_receivers;i++){
			for(j=0;j<recvcount;j++){
				if(receive_buff[(i-n_senders)*recvcount+j]!=MAP_DATA(i,j)){
					printf("rank %d: test failed at i=%d,j=%d,%d!=%d\n",rank,i,j,receive_buff[(i-n_senders)*recvcount+j],MAP_DATA(i,j));
				}
			}
                }
        }else{
		for(i=0;i<n_senders;i++){
			for(j=0;j<recvcount;j++){
				if(receive_buff[i*recvcount+j]!=MAP_DATA(i,j)){
					printf("rank %d: test failed at i=%d,j=%d,%d!=%d\n",rank,i,j,receive_buff[i*recvcount+j],MAP_DATA(i,j));
				}
			}
                }
	}
}

/*
  Internal function of bipartite_allgatherv_universal_full_duplex_emulation.
  This function computes the message size to be sent to the remote group for this process.
*/
void send_map(int rank,int n_senders,int* size1, int *size2,int local_rank,int local_size,int remote_size,int* sendcounts,int* send_ranks,int* rank_size){
	int temp,message_group_size,i,sendcount,temp2=0,high_stacks,send_start,sendcount_total;
	temp=0;
	for(i=0;i<local_size;i++){
		if(local_rank==i){
			temp2=temp;
		}
		if(rank<n_senders){
			temp+=size1[i];
		}else{
			temp+=size2[i];
		}
	}
	if(rank<n_senders){
		sendcount_total=size1[local_rank];
	}else{
		sendcount_total=size2[local_rank];
	}
	message_group_size=temp/remote_size;
	high_stacks=temp%remote_size-remote_size*((temp%remote_size)/remote_size);
	if(temp2<high_stacks*(message_group_size+1)){
		send_start=temp2/(message_group_size+1);
		//sendcount=message_group_size+1-(temp2-(message_group_size+1)*send_start);
		sendcount=(message_group_size+1)*(send_start+1)-temp2;
	}else{
		send_start=high_stacks+(temp2-high_stacks*(message_group_size+1))/message_group_size;
		//sendcount=message_group_size-(temp2-(message_group_size+1)*high_stacks-message_group_size*(send_start-high_stacks));
		sendcount=(message_group_size+1)*high_stacks+message_group_size*(send_start-high_stacks+1)-temp2;
	}
	for(i=send_start;i<remote_size&&sendcount_total>0;i++){
		if(i>send_start){
			if(i<high_stacks){
				sendcount=message_group_size+1;
			}else{
				sendcount=message_group_size;
			}
		}
		if(sendcount_total>sendcount){
			sendcounts[i-send_start]=sendcount;
			sendcount_total-=sendcount;
		}else{
			sendcounts[i-send_start]=sendcount_total;
			sendcount_total=0;
		}
		if(rank<n_senders){
			send_ranks[i-send_start]=i+n_senders;
		}else{
			send_ranks[i-send_start]=i;
		}
	}
	rank_size[0]=i-send_start;
}
/*
  Internal function of bipartite_allgatherv_universal_full_duplex_emulation.
  This function computes the message size to be received from the remote group for this process.
*/
void receive_map(int rank,int n_senders,int* size1, int *size2,int local_rank,int local_size,int remote_size,int* recvcounts,int* recv_ranks,int* rank_size,int *total_recv_size){
	int temp,message_group_size,i,sendcount,temp2,high_stacks,message_rank,recvcount;
	temp=0;
	for(i=0;i<remote_size;i++){
		if(rank<n_senders){
			temp+=size2[i];
		}else{
			temp+=size1[i];
		}
	}
	total_recv_size[0]=temp;
	message_group_size=temp/local_size;
	high_stacks=temp%local_size-local_size*((temp%local_size)/local_size);
	temp2=0;
	message_rank=0;
	rank_size[0]=0;
	if(rank<n_senders){
		sendcount=size2[message_rank];
	}else{
		sendcount=size1[message_rank];
	}
	for(i=0;i<=local_rank;i++){
		if(i<high_stacks){
			recvcount=message_group_size+1;
		}else{
			recvcount=message_group_size;
		}
		while(recvcount>0){
			if(i==local_rank){
				if(rank<n_senders){
					recv_ranks[rank_size[0]]=message_rank+n_senders;
				}else{
					recv_ranks[rank_size[0]]=message_rank;
				}
			}
			if(sendcount<=recvcount){
				recvcount-=sendcount;
				if(i==local_rank){
					recvcounts[rank_size[0]]=sendcount;
				}
				message_rank++;
				if(rank<n_senders){
					sendcount=size2[message_rank];
				}else{
					sendcount=size1[message_rank];
				}
			}else{
				if(i==local_rank){
					recvcounts[rank_size[0]]=recvcount;
				}
				sendcount-=recvcount;
				recvcount=0;
			}
			if(i==local_rank){
				rank_size[0]++;
			}
		}
	}
}

/*
  Experiments with Algorithm 2 of the paper submitted to journal of parallel computing.
  All cases of process size and message size are supported.
    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. v_size1: an array of number of dim_x elements in group A per process
    6. v_size2: an array of number of dim_x elements in group B per process
*/
double* bipartite_allgatherv_universal_full_duplex_emulation(int rank,int dim_x,int n_senders,int n_receivers,int* v_size1, int *v_size2){
	void *tmp_buf=NULL,*tmp_buf_ptr,*original_ptr,*recvbuf,*sendbuf,*send_buf_tmp;
	int *receive_buff=NULL,*send_buff=NULL;
	MPI_Status *status;
        double allgather_time=0,start,send_size_time,map_time;
	int remote_size=0,local_size=0,local_rank,i,message_group_size,high_stacks,send_rank_size,recv_rank_size,total_recv_size,tmp_buf_size,target_rank,message_rank,color;
	MPI_Comm new_comm;
	int *displs,*recvcounts2;
	//Determine which one is low group
	int size1[n_senders];
	int size2[n_receivers];
	MPI_Request *request;
	memcpy(size1,v_size1,sizeof(int)*n_senders);
	memcpy(size2,v_size2,sizeof(int)*n_receivers);
	if(rank<n_senders){
		local_rank=rank;
		remote_size=n_receivers;
		local_size=n_senders;
	}else{
		local_rank=rank-n_senders;
		remote_size=n_senders;
		local_size=n_receivers;
	}
	int sendcounts[remote_size];
	int send_ranks[remote_size];
	int recvcounts[remote_size];
	int recv_ranks[remote_size];
	int dummy_buf[remote_size+local_size];
	double *result=(double*)malloc(3*sizeof(double));

        start=MPI_Wtime();
	//WORLD group
        if (rank<n_senders){
            color = 0;
        }else{
            color = 1;
        }
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
	//Data creations
	create_vector_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,size1,size2,dim_x);
        start=MPI_Wtime()-start;
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;

        send_size_time=MPI_Wtime();
	//No effect, simply simulate the synchronization of all send size.
	MPI_Allgather(size1,1,MPI_INT,dummy_buf,1,MPI_INT,MPI_COMM_WORLD);
	send_size_time=MPI_Wtime()-send_size_time;

        map_time=MPI_Wtime();
	send_map(rank,n_senders,size1,size2,local_rank,local_size,remote_size,sendcounts,send_ranks,&send_rank_size);
	receive_map(rank,n_senders,size1,size2,local_rank,local_size,remote_size,recvcounts,recv_ranks,&recv_rank_size,&total_recv_size);
	high_stacks=total_recv_size%local_size-local_size*((total_recv_size%local_size)/local_size);
	message_group_size=total_recv_size/local_size;
	tmp_buf_size=message_group_size+1;
	if(tmp_buf_size>0){
		tmp_buf=malloc(sizeof(int)*tmp_buf_size);
	}
	tmp_buf_ptr=tmp_buf;
	original_ptr=tmp_buf;
	//Emulation for full-duplex comm algorithm
	target_rank=0;
	message_rank=0;
	send_buf_tmp=sendbuf;
        request = (MPI_Request*) malloc(sizeof(MPI_Request)*(send_rank_size+recv_rank_size));
        status = (MPI_Status*) malloc(sizeof(MPI_Status)*(send_rank_size+recv_rank_size));
	for(i=0;i<send_rank_size;i++){
		MPI_Isend(send_buf_tmp, sendcounts[i], MPI_INT,
			send_ranks[i], 0,
			MPI_COMM_WORLD,request+i);
		send_buf_tmp+=sizeof(int)*sendcounts[i];
	}
	for(i=0;i<recv_rank_size;i++){
		MPI_Irecv(tmp_buf_ptr, recvcounts[i], MPI_INT,
			 recv_ranks[i], 0,
            		 MPI_COMM_WORLD, request+send_rank_size+i);
		tmp_buf_ptr+=sizeof(int)*recvcounts[i];
	}

	displs=(int*)malloc(local_size*sizeof(int));
	recvcounts2=(int*)malloc(local_size*sizeof(int));
	for(i=0;i<local_size;i++){
		if(i<high_stacks){
			recvcounts2[i]=message_group_size+1;
		}else{
			recvcounts2[i]=message_group_size;
		}
		if(i==0){
			displs[i]=0;
		}else{
			displs[i]=displs[i-1]+recvcounts2[i-1];
		}
	}
	if(local_rank<high_stacks){
		message_group_size++;
	}

        MPI_Waitall(send_rank_size+recv_rank_size, request, status);

	map_time=MPI_Wtime()-map_time;

	allgather_time=MPI_Wtime();
	MPI_Allgatherv(tmp_buf,message_group_size,MPI_INT,recvbuf,recvcounts2,displs,MPI_INT,new_comm);
	allgather_time=MPI_Wtime()-allgather_time;
	MPI_Reduce(&send_size_time, result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&map_time, result+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&allgather_time, result+2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	#if DEBUG==1
	if(rank<n_senders){
		validate_vector_result(rank,n_senders,n_receivers,receive_buff,size2);
	}else{
		validate_vector_result(rank,n_senders,n_receivers,receive_buff,size1);
	}
	#endif
	if(rank==0){
		printf("Universal full duplex allgatherv time=%lf(message size time)+%lf(intergroup message transfer)+%lf(intragroup allgather)=%lf\n",result[0],result[1],result[2],result[0]+result[1]+result[2]);
	}
	MPI_Comm_free(&new_comm);
	free(recvcounts2);
	free(displs);
	free(original_ptr);
	free(sendbuf);
	free(recvbuf);
        free(request);
        free(status);
	return result;
}
/*
  Experiments with default MPI library for intergroup Allgatherv.
  We time the execution time of MPI_Allgather only. (assuming intergroup communicator is available at the beginning.)
    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. v_size1: an array of number of dim_x elements in group A per process
    6. v_size2: an array of number of dim_x elements in group B per process
*/
double bipartite_allgatherv_full_duplex(int rank,int dim_x,int n_senders,int n_receivers,int* v_size1, int *v_size2){
	//local variables
	void *sendbuf=NULL, *recvbuf=NULL;
	int color, sendcount, *recvcounts,local_size,i;
        double allgather_time=0,start,total_time;
	MPI_Comm local_comm, remote_comm;
	int *receive_buff=NULL,*send_buff=NULL;
	int size1[n_senders];
	int size2[n_receivers];
	memcpy(size1,v_size1,sizeof(int)*n_senders);
	memcpy(size2,v_size2,sizeof(int)*n_receivers);
        start=MPI_Wtime();
	//WORLD group
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	/* split MPI processes into two groups */
	color = (rank < n_senders) ? 0 : 1;
	err = MPI_Comm_split(MPI_COMM_WORLD, color, rank, &local_comm); ERR
	/* obtain the inter communicator from the other group */
	err = MPI_Intercomm_create(local_comm, 0, MPI_COMM_WORLD,
                               (color == 0) ? n_senders : 0, 1, &remote_comm); ERR
	/* only the processes in 1st group send and only 2nd group receive */
	//printf("rank=%d\n",rank);
	create_vector_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,size1,size2,dim_x);
	if(rank<n_senders){
		sendcount=size1[rank];
		recvcounts=size2;
	}else{
		sendcount=size2[rank-n_senders];
		recvcounts=size1;
	}
	if(rank<n_senders){
		local_size=n_receivers;
	}else{
		local_size=n_senders;
	}
	int* displs=(int*)malloc(local_size*sizeof(int));
	for(i=0;i<local_size;i++){
		if(i==0){
			displs[i]=0;
		}else{
			displs[i]=displs[i-1]+recvcounts[i-1];
		}
	}
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
        total_time=MPI_Wtime();
        start=MPI_Wtime();
	/* use Allgather to achieve effect of all Bcast from the 1st group * to 2nd group */
	err = MPI_Allgatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs, MPI_INT,remote_comm); ERR\
	//Timing
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	double result;
	MPI_Reduce(&allgather_time, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//Check if receivers actually received results.
	#if DEBUG==1
	validate_vector_result(rank,n_senders,n_receivers,receive_buff,recvcounts);
	#endif
	if(rank==0){
		printf("Benchmark full duplex allgatherv time=%lf\n",result);
	}
	MPI_Comm_free(&remote_comm);
	MPI_Comm_free(&local_comm);
	free(sendbuf);
	free(recvbuf);
	free(displs);
	return result;
}
/*
  Experiments with Algorithm 1 of the paper submitted to journal of parallel computing.
  This algorithm simplifies the algorithm adopted by bipartite_allgather_universal_full_duplex_emulation.
  All cases of process size and message size are supported.
    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. size1: number of dim_x elements in group A
    6. size2: number of dim_x elements in group B
*/
double bipartite_allgather_simple_universal_full_duplex_emulation(int rank,int dim_x,int n_senders,int n_receivers,int size1, int size2){
	//local variables
	void *tmp_buf=NULL,*tmp_buf_ptr,*original_ptr,*recvbuf,*sendbuf,*adjusted_sendbuf;
	int sendcount, recvcount,i,color;
        double allgather_time=0,setup_time=0,start,total_time;
	int remote_size=0,local_size=0,tmp_buf_size=0,local_rank,local_group_number=0,temp,temp2=0,target_rank=0,remainder2=0,shift=0;
	MPI_Status status;
	int *displs,*recvcounts;
	int adjusted_sendcount,adjusted_recvcount=0,remainder;
	MPI_Comm new_comm;
	int *receive_buff=NULL,*send_buff=NULL;
	//Determine which one is low group
	if(rank<n_senders){
		local_rank=rank;
		remote_size=n_receivers;
		local_size=n_senders;
		color=1;
	}else{
		local_rank=rank-n_senders;
		remote_size=n_senders;
		local_size=n_receivers;
		color=0;
	}
        start=MPI_Wtime();
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
	//Data creations
	create_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,&sendcount,&recvcount,dim_x,size1,size2);
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
        start=MPI_Wtime();
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
	if( remote_size >= local_size ){
		remainder = remote_size % local_size;
		temp = (remote_size + local_size - 1) / local_size;
		temp2 = remote_size / local_size;
		if (local_rank < remainder){
			local_group_number = temp;
			shift = local_rank * temp;
		}else{
			local_group_number = temp2;
			shift = temp * remainder + (local_rank-remainder) * temp2;
		}
		tmp_buf_size = local_group_number * recvcount;
	}else{
		remainder = local_size % remote_size;
		temp = (local_size + remote_size - 1) / remote_size;
		temp2 = local_size / remote_size;
		if ( local_rank < remainder * temp ){
			target_rank = local_rank / temp;
			remainder2 = recvcount % temp;
			if (local_rank % temp < remainder2){
				adjusted_recvcount = ( recvcount + temp -1 ) / temp;
			} else{
				adjusted_recvcount = recvcount / temp;
                        }
		} else{
			target_rank = remainder + (local_rank - remainder * temp) / temp2;
			remainder2 = recvcount % temp2;
			if ((local_rank-remainder*temp) % temp2 < remainder2){
				adjusted_recvcount = ( recvcount + temp2 -1 ) / temp2;
			} else{
				adjusted_recvcount = recvcount / temp2;
                        }
		}
		tmp_buf_size=adjusted_recvcount;
	}
	if(tmp_buf_size>0){
		tmp_buf=malloc(sizeof(int)*tmp_buf_size);
	}
	tmp_buf_ptr=tmp_buf;
	original_ptr=tmp_buf;
	displs = (int*) malloc(sizeof(int)*local_size);
	recvcounts = (int*) malloc(sizeof(int)*local_size);
	if ( remote_size < local_size ){
		if (rank < n_senders){
			target_rank += n_senders;
		}
		//printf("rank %d, temp = %d, temp2=%d, adjusted_recvcount=%d\n",rank,temp,temp2,adjusted_recvcount);
		MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
			target_rank, 0,
			tmp_buf_ptr, adjusted_recvcount, MPI_INT,
			target_rank, 0,
			MPI_COMM_WORLD, &status);
		displs[0] = 0;
		for ( i = 0; i < local_size; i++){
			if ( i < remainder * temp  ){
				remainder2 = recvcount % temp;
				if (i % temp < remainder2){
					recvcounts[i] = ( recvcount + temp -1 ) / temp;
				} else{
					recvcounts[i] = recvcount / temp;
		                }
			} else{
				remainder2 = recvcount % temp2;
				if ((i-remainder*temp) % temp2 < remainder2){
					recvcounts[i] = ( recvcount + temp2 -1 ) / temp2;
				} else{
					recvcounts[i] = recvcount / temp2;
		                }
			}
			if ( i > 0 ){
				displs[i] = displs[i-1] + recvcounts[i-1];
			}
			//printf("rank %d, displs[%d]=%d, recvcounts[%d]=%d\n",rank,i,displs[i],i,recvcounts[i]);
		}
		MPI_Allgatherv(tmp_buf,adjusted_recvcount,MPI_INT,recvbuf,recvcounts,displs,MPI_INT,new_comm);
	} else{
		adjusted_sendbuf = sendbuf;
		remainder2 = sendcount % local_group_number;
		//printf("rank %d, temp = %d, temp2=%d, shift=%d, local_group_number=%d\n",rank,temp,temp2,shift,local_group_number);
		for ( i = 0; i < local_group_number; i++){
			target_rank = shift + i;
			if (rank < n_senders){
				target_rank += n_senders;
			}
			if ( i < remainder2 ){
				adjusted_sendcount = (sendcount + local_group_number - 1 ) / local_group_number;
			}else{
				adjusted_sendcount = sendcount / local_group_number;
			}

			MPI_Sendrecv(adjusted_sendbuf, adjusted_sendcount, MPI_INT,
				target_rank, 0,
				tmp_buf_ptr, recvcount, MPI_INT,
				target_rank, 0,
				MPI_COMM_WORLD, &status);
			tmp_buf_ptr += recvcount * sizeof(int);
			adjusted_sendbuf += adjusted_sendcount * sizeof(int);
		}
		displs[0] = 0;
		for ( i = 0; i < local_size; i++){
			if ( i < remainder ){
				recvcounts[i] = recvcount * temp;
			} else{
				recvcounts[i] = recvcount * temp2;
			}
			if ( i > 0 ){
				displs[i] = displs[i-1] + recvcounts[i-1];
			}
			//printf("rank %d, displs[%d]=%d, recvcounts[%d]=%d\n",rank,i,displs[i],i,recvcounts[i]);
		}
		MPI_Allgatherv(tmp_buf,recvcounts[local_rank],MPI_INT,recvbuf,recvcounts,displs,MPI_INT,new_comm);
	}
	free(displs);
	free(recvcounts);
	if(tmp_buf_size>0){
		free(original_ptr);
	}
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	double result;
	MPI_Reduce(&allgather_time, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//Check if receivers actually received results.
	#if DEBUG==1
	validate_result(rank,n_senders,n_receivers,receive_buff,recvcount);
	#endif
	if(rank==0){
		printf("Full-duplex simple universal Allgather emulation setup=%lf,processing=%lf\n",setup_time,result);
	}
	free(sendbuf);
	free(recvbuf);
	MPI_Comm_free(&new_comm);
	return result;
}
/*
  Experiments with Algorithm 2 of Euro MPI paper.
  This algorithm handles the case where message size is smaller than the number of process case in a complicated way (as mentioned in the paper).
  Reference: Kang, Qiao, Jesper Larsson Träff, Reda Al-Bahrani, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao. "Full-duplex inter-group all-to-all broadcast algorithms with optimal bandwidth." In Proceedings of the 25th European MPI Users’ Group Meeting, p. 1. 2018.

    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. size1: number of dim_x elements in group A
    6. size2: number of dim_x elements in group B
*/
double bipartite_allgather_universal_full_duplex_emulation(int rank,int dim_x,int n_senders,int n_receivers,int size1, int size2){
	//local variables
	void *tmp_buf=NULL,*tmp_buf_ptr,*original_ptr,*recvbuf,*sendbuf,*adjusted_sendbuf;
	int sendcount, recvcount,i;
        double allgather_time=0,setup_time=0,start,total_time;
	int group_number=0,remote_size=0,local_size=0,tmp_buf_size=0,local_rank,local_group_number,temp,temp2=0,message_rank,target_rank,remainders2=0,shift;
	MPI_Status status;
	int *subgroup_ranks,*displs,*recvcounts;
	int adjusted_sendcount,adjusted_recvcount,tmp_count,message_group_size=0,remainders,message_group_number=0,subgroup_size;
	MPI_Group world_group,subgroup,subgroup2;
	MPI_Comm new_comm,new_comm2;
	int *receive_buff=NULL,*send_buff=NULL;
	//Determine which one is low group
	if(rank<n_senders){
		local_rank=rank;
		remote_size=n_receivers;
		local_size=n_senders;
	}else{
		local_rank=rank-n_senders;
		remote_size=n_senders;
		local_size=n_receivers;
	}
        start=MPI_Wtime();
	//WORLD group
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	//Data creations
	create_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,&sendcount,&recvcount,dim_x,size1,size2);
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
        start=MPI_Wtime();
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
	if(remote_size>local_size){
		group_number=remote_size/local_size;
		tmp_buf_size=((remote_size+local_size-1)/local_size)*recvcount;
	}else{
		group_number=local_size/remote_size;
		//Adjusted group count (for fragmented message) + recvbuf size + 2 times of recvcount (for padding purpose).
		tmp_buf_size=(1+2*group_number)*((recvcount+group_number-1)/group_number);
	}
	if(tmp_buf_size>0){
		tmp_buf=malloc(sizeof(int)*tmp_buf_size);
	}
	tmp_buf_ptr=tmp_buf;
	original_ptr=tmp_buf;
	//Emulation for full-duplex comm algorithm
	if(remote_size<local_size){
		//message size divided by group number, taking ceiling
		adjusted_recvcount=(recvcount+group_number-1)/group_number;
		tmp_buf+=sizeof(int)*adjusted_recvcount;
		local_group_number=local_rank/remote_size;

		remainders=local_size%remote_size;
		//Lowest upper bound for all high order stacks.
		temp=(remainders-group_number*(remainders/group_number))*(remote_size+remainders/group_number+1);
		//Sending to remote group in order
		temp2=(remainders-remote_size*(remainders/remote_size))*(group_number+remainders/remote_size+1);
		if(local_rank<temp2){
			target_rank=local_rank/(group_number+remainders/remote_size+1);
		}else{
			target_rank=remainders-remote_size*(remainders/remote_size)+(local_rank-temp2)/(group_number+remainders/remote_size);
		}
		//Padding ranks for sending using world communicator.
		if(rank<n_senders){
			target_rank+=n_senders;
		}
		if(group_number>=recvcount){
			//Remainder group at rank group and recvcount level
			remainders2=group_number%recvcount;
			//How many groups of size recvcount we want to have.
			message_group_number=group_number/recvcount;
			//Low stack size for rank groups.
			message_group_size=recvcount+remainders2/message_group_number;
			//Lowest upper bound for group level high stack ranks.
			temp2=(remainders2-message_group_number*(remainders2/message_group_number))*(message_group_size+1);
		}
		if(local_rank<temp){
			local_group_number=local_rank/(remote_size+remainders/group_number+1);
		}else{
			local_group_number=(remainders-group_number*(remainders/group_number))+(local_rank-temp)/(remote_size+remainders/group_number);
		}
		//If these are remainder processes, in any level senses.
		if((local_rank<temp&&local_rank%(remote_size+remainders/group_number+1)>=remote_size)
		||(local_rank>=temp&&(local_rank-temp)%(remote_size+remainders/group_number)>=remote_size)
		){
			//Should not receive anything for remainder processes.
			MPI_Send(sendbuf, sendcount, MPI_INT,
				target_rank, 0,
				MPI_COMM_WORLD);
		}else{
			//In case of large messages.
			if(recvcount>=group_number){
				//Check if the sender of remote group run out of messages due to ceiling of message size.
				if(adjusted_recvcount*(local_group_number+1)>recvcount&&adjusted_recvcount*local_group_number<=recvcount){
					//The last partial message
					adjusted_recvcount=recvcount-adjusted_recvcount*local_group_number;
				}else if(adjusted_recvcount*local_group_number>recvcount){
					//No more messages to be received.
					adjusted_recvcount=0;
				}
			}
			//Determine the rank for receiving using high/low stack ordering.
			//Receiving is modulus pair.
			if(local_rank<temp){
				//Decide location in a high stack.
				message_rank=local_rank%(remote_size+remainders/group_number+1);
			}else{
				//Decide location in a low stack.
				message_rank=(local_rank-temp)%(remote_size+remainders/group_number);
			}

			if(rank<n_senders){
				message_rank+=n_senders;
			}

			if(recvcount<group_number&&((local_group_number<temp2&&local_group_number%(message_group_size+1)>=recvcount)||(local_group_number>=temp2&&(local_group_number-temp2)%message_group_size>=recvcount))){
				MPI_Send(sendbuf, sendcount, MPI_INT,
					target_rank, 0,
					MPI_COMM_WORLD);
			}else{
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					target_rank, 0,
					tmp_buf_ptr, adjusted_recvcount, MPI_INT,
					message_rank, 0,
					MPI_COMM_WORLD, &status);
			}
			//Use fragmented group size to perform allgather, ignoring remainders.
			if(recvcount>=group_number){
				//Message size larger than group number. Remote group size is divisible for total non-remainder ranks.
				//Assign interleaved groups.
				message_group_size=group_number;
				subgroup_size=group_number;
				subgroup_ranks=(int*)malloc((message_group_size)*sizeof(int));
				if(local_rank<temp){
					shift=local_rank%(remote_size+remainders/group_number+1);
				}else{
					shift=(local_rank-temp)%(remote_size+remainders/group_number);
				}
				for(i=0;i<message_group_size;i++){
					//Divide groups of ranks into high/low stacks.
					if(i<remainders-group_number*(remainders/group_number)){
						subgroup_ranks[i]=i*(remote_size+remainders/group_number+1)+shift;
					}else{
						subgroup_ranks[i]=temp+(i-remainders+group_number*(remainders/group_number))*(remote_size+remainders/group_number)+shift;
					}
					//Padding for world rank.
					if(rank>=n_senders){
						subgroup_ranks[i]+=n_senders;
					}
				}
				//printf("rank=%d,message_group_size=%d\n",local_rank,message_group_size);
				MPI_Group_incl(world_group, message_group_size, subgroup_ranks, &subgroup2);
			}else{
				//Message size smaller than group number. Need crawling of data.
				subgroup_size=message_group_size;
				//Fold actual rank to index of group ranks. Compute shift for actual ranks notation for the group.
				if(local_rank<temp){
					message_rank=local_rank/(remote_size+remainders/group_number+1);
					shift=local_rank%(remote_size+remainders/group_number+1);
				}else{
					//Number of high stacks + which index of low stack the rank is at.
					message_rank=(remainders-group_number*(remainders/group_number))+(local_rank-temp)/(remote_size+remainders/group_number);
					shift=(local_rank-temp)%(remote_size+remainders/group_number);
				}
				//How many high stacks we have (at group index level).
				temp2=remainders2-message_group_number*(remainders2/message_group_number);
				//Fold index of group ranks to index of subgroup ranks.
				if(message_rank<temp2*(message_group_size+1)){
					subgroup_size++;
					message_rank=message_rank/(message_group_size+1);
				}else{
					message_rank=temp2+(message_rank-temp2*(message_group_size+1))/message_group_size;
				}
				subgroup_ranks=(int*)malloc(sizeof(int)*subgroup_size);
				for(i=0;i<subgroup_size;i++){
					//Unfold to index of rank group, i.e the local group number of this rank. (this tells which of the rank group the target is at)
					if(message_rank<temp2){
						target_rank=message_rank*(message_group_size+1)+i;
					}else{
						target_rank=temp2*(message_group_size+1)+(message_rank-temp2)*message_group_size+i;
					}
					//Unfold to actual ranks by shifting all the previous stacks (this tells the actual rank of the beginning of the rank group that the target is at).
					if(target_rank<remainders-group_number*(remainders/group_number)){
						target_rank=target_rank*(remote_size+remainders/group_number+1)+shift;
					}else{
						target_rank=temp+(target_rank-remainders+group_number*(remainders/group_number))*(remote_size+remainders/group_number)+shift;
					}
					//Shift final rank to the correct displacement.
					if(rank<n_senders){
						subgroup_ranks[i]=target_rank;				
					}else{
						subgroup_ranks[i]=target_rank+n_senders;
					}
				}
				MPI_Group_incl(world_group, subgroup_size, subgroup_ranks, &subgroup2);
			}
			MPI_Comm_create_group(MPI_COMM_WORLD, subgroup2, 0, &new_comm2);

			//Allgather for fragmented messages (ceiling of fragmented message size over max(group_number,recvcount))	
			MPI_Allgather ( 
				tmp_buf_ptr,
				(recvcount+group_number-1)/group_number,
				MPI_INT,
				tmp_buf,
				(recvcount+group_number-1)/group_number,
				MPI_INT,
				new_comm2);
			MPI_Group_free(&subgroup2);
			MPI_Comm_free(&new_comm2);
			free(subgroup_ranks);
		}
		tmp_buf_ptr=tmp_buf;
		tmp_buf=original_ptr;
		message_group_size=remote_size+remainders/group_number;
		if(local_rank<temp){
			if(local_rank%(remote_size+remainders/group_number+1)<remote_size){
				adjusted_recvcount=recvcount;
			}else{
				adjusted_recvcount=0;
			}
			message_group_size++;
		}else{
			if((local_rank-temp)%(remote_size+remainders/group_number)<remote_size){
				adjusted_recvcount=recvcount;
			}else{
				adjusted_recvcount=0;
			}
		}
		subgroup_ranks=(int*)malloc(sizeof(int)*message_group_size);
		displs=(int*)malloc(sizeof(int)*message_group_size);
		recvcounts=(int*)malloc(sizeof(int)*message_group_size);
		for(i=0;i<message_group_size;i++){
			if(local_rank<temp){
				//At high stacks (with extra element).
				//Find the group number and load the previous stacks as base.
				subgroup_ranks[i]=(local_rank/(remote_size+remainders/group_number+1))*(remote_size+remainders/group_number+1)+i;
			}else{
				//At low stacks
				//Shift high stack size (temp) first and load the rest of low stacks.
				subgroup_ranks[i]=temp+((local_rank-temp)/((remote_size+remainders/group_number)))*(remote_size+remainders/group_number)+i;
			}
			//if(local_rank==0){
				//printf("rank=%d,subgroup_ranks[%d]=%d\n",local_rank,i,subgroup_ranks[i]);
			//}
			displs[i]=i*recvcount;
			if(i<remote_size){
				recvcounts[i]=recvcount;		
			}else{
				recvcounts[i]=0;
			}
			//printf("rank %d,displs[%d]=%d,recvcounts[%d]=%d\n",local_rank,i,displs[i],i,recvcounts[i]);
			if(rank>=n_senders){
				subgroup_ranks[i]+=n_senders;
			}
		}
		//Join remainder processes to final allgather groups.
		MPI_Group_incl(world_group, message_group_size, subgroup_ranks, &subgroup);
		MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &new_comm);
		//printf("rank=%d,value=%d,%d\n",rank,(int)((int*)tmp_buf_ptr),(int)((int*)tmp_buf));
		MPI_Allgatherv(tmp_buf_ptr,adjusted_recvcount,MPI_INT,recvbuf,recvcounts,displs,MPI_INT,new_comm);
		free(displs);
		free(recvcounts);
		free(subgroup_ranks);
		MPI_Group_free(&subgroup);
		MPI_Comm_free(&new_comm);
	}else{
		adjusted_sendcount=(sendcount+group_number-1)/group_number;
		adjusted_sendbuf=sendbuf;
		//size of remainder group
		remainders=remote_size%local_size;
		//low stack size at remote site.
		temp=(remainders-group_number*(remainders/group_number));
		temp2=(remainders-local_size*(remainders/local_size));
		//assign local_size number of continuous groups. Figure out the first index.
		if(local_rank<temp2){
			shift=local_rank*(group_number+remainders/local_size+1);
		}else{
			shift=temp2*(group_number+remainders/local_size+1)+(local_rank-temp2)*(group_number+remainders/local_size);
		}
		if(sendcount>=group_number){
			//For large message size
			for(i=0;i<group_number;i++){
				//need to figure out how many messages to be sent (if we run out of messages)
				if(adjusted_sendcount*(i+1)>sendcount&&adjusted_sendcount*i<=sendcount){
					tmp_count=sendcount-adjusted_sendcount*i;
				}else if(adjusted_sendcount*i>sendcount){
					tmp_count=0;
				}else{
					tmp_count=adjusted_sendcount;
				}
				if(i<temp){
					target_rank=i*(local_size+remainders/group_number+1)+local_rank;
				}else{
					target_rank=temp*(local_size+remainders/group_number+1)+(i-temp)*(local_size+remainders/group_number)+local_rank;
				}
				message_rank=shift+i;

				if(rank<n_senders){
					target_rank+=n_senders;
					message_rank+=n_senders;
				}
				//printf("A rank %d sending to %d with %d messages\n",local_rank,target_rank,tmp_count);

				MPI_Sendrecv(adjusted_sendbuf, tmp_count, MPI_INT,
					target_rank, 0,
					tmp_buf_ptr, recvcount, MPI_INT,
					message_rank, 0,
					MPI_COMM_WORLD, &status);

				//printf("B rank %d sending to %d\n",local_rank,target_rank);
				tmp_buf_ptr+=recvcount*sizeof(int);
				adjusted_sendbuf+=adjusted_sendcount*sizeof(int);
			}
		}else{
			//For small message
			//Remainder group at rank group and recvcount level
			remainders2=group_number%sendcount;
			//How many groups of size sendcount we want to have.
			message_group_number=group_number/sendcount;
			//Low stack size for rank groups (at group index level).
			message_group_size=sendcount+remainders2/message_group_number;
			//Lowest upper bound for high stack ranks (at group index level).
			temp2=(remainders2-message_group_number*(remainders2/message_group_number))*(message_group_size+1);
			//Send only group_number-remainders2 number of times.
			subgroup_size=0;
			for(i=0;i<group_number;i++){
				//Which receiver?
				message_rank=shift+i;
				if(rank<n_senders){
					message_rank+=n_senders;
				}
				//Jump the remainder rank groups at group index level.
				if((i<temp2&&i%(message_group_size+1)>=sendcount)||(i>=temp2&&(i-temp2)%message_group_size>=sendcount)){
					//No need to send anything, use the remainder mechanism at the remote site to handle the rest!
					//printf("rank=%d, ignoring %d\n",local_rank,target_rank);
					MPI_Recv(tmp_buf_ptr, recvcount, MPI_INT,
					message_rank, 0,
					MPI_COMM_WORLD, &status);
				}else{
					//Use group number and local rank to unfold the actual rank.
					if(i<temp){
						target_rank=i*(local_size+remainders/group_number+1)+local_rank;
					}else{
						target_rank=temp*(local_size+remainders/group_number+1)+(i-temp)*(local_size+remainders/group_number)+local_rank;
					}
					if(rank<n_senders){
						target_rank+=n_senders;
					}
					MPI_Sendrecv(adjusted_sendbuf, adjusted_sendcount, MPI_INT,
						target_rank, 0,
						tmp_buf_ptr, recvcount, MPI_INT,
						message_rank, 0,
						MPI_COMM_WORLD, &status);
					subgroup_size++;
					//Rotate the message sequence when the end is reached.
					if(subgroup_size%sendcount==0){
						adjusted_sendbuf=sendbuf;
					}else{
						adjusted_sendbuf+=adjusted_sendcount*sizeof(int);
					}
				}
				tmp_buf_ptr+=recvcount*sizeof(int);
			}
		}
		temp2=(remainders-local_size*(remainders/local_size));
		if(local_rank<temp2){
			message_group_size=group_number+remainders/local_size+1;
		}else{
			message_group_size=group_number+remainders/local_size;
		}
		//Keep receiving from remainder group (one of local_size number of group defined by temp2 stack lower bound) the remote site
		for(i=group_number;i<message_group_size;i++){
			message_rank=shift+i;
			if(rank<n_senders){
				message_rank+=n_senders;
			}
			MPI_Recv(tmp_buf_ptr, recvcount, MPI_INT,
					message_rank, 0,
					MPI_COMM_WORLD, &status);
			tmp_buf_ptr+=recvcount*sizeof(int);
		}
		subgroup_ranks=(int*)malloc(local_size*sizeof(int));
		displs=(int*)malloc(local_size*sizeof(int));
		recvcounts=(int*)malloc(local_size*sizeof(int));
		for(i=0;i<local_size;i++){
			if(i<temp2){
				displs[i]=recvcount*i*(group_number+remainders/local_size+1);
				recvcounts[i]=recvcount*(group_number+remainders/local_size+1);
			}else{
				displs[i]=recvcount*(temp2*(group_number+remainders/local_size+1)+(i-temp2)*(group_number+remainders/local_size));
				recvcounts[i]=recvcount*(group_number+remainders/local_size);
			}
			//printf("rank %d,displs[%d]=%d,recvcounts[%d]=%d\n",local_rank,i,displs[i],i,recvcounts[i]);
			if(rank<n_senders){
				subgroup_ranks[i]=i;
			}else{
				subgroup_ranks[i]=n_senders+i;
			}
		}
		MPI_Group_incl(world_group, local_size, subgroup_ranks, &subgroup);
		MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &new_comm);
		MPI_Allgatherv(tmp_buf,recvcounts[local_rank],MPI_INT,recvbuf,recvcounts,displs,MPI_INT,new_comm);
		free(displs);
		free(recvcounts);
		free(subgroup_ranks);
		MPI_Group_free(&subgroup);
		MPI_Comm_free(&new_comm);
	}
	if(tmp_buf_size>0){
		free(original_ptr);
	}
	//Timing
	//printf("rank=%d\n",rank);
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	double result;
	MPI_Reduce(&allgather_time, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//Check if receivers actually received results.
	#if DEBUG==1
	validate_result(rank,n_senders,n_receivers,receive_buff,recvcount);
	#endif
	if(rank==0){
		printf("Full-duplex universal Allgather emulation setup=%lf,processing=%lf\n",setup_time,result);
	}
	free(sendbuf);
	free(recvbuf);
	return result;
}

/*
  Experiments with Algorithm 1 of Euro MPI paper.
  This algorithm suffers from message hazard mentioned in the paper.
  Reference: Kang, Qiao, Jesper Larsson Träff, Reda Al-Bahrani, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao. "Full-duplex inter-group all-to-all broadcast algorithms with optimal bandwidth." In Proceedings of the 25th European MPI Users’ Group Meeting, p. 1. 2018.

    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. size1: number of dim_x elements in group A
    6. size2: number of dim_x elements in group B
*/
void bipartite_allgather_full_duplex_emulation(int rank,int dim_x,int n_senders,int n_receivers,int size1, int size2){
	//local variables
	void *tmp_buf=NULL,*tmp_buf_ptr,*recvbuf,*sendbuf;
	int sendcount, recvcount,i;
        double allgather_time=0,setup_time=0,start,total_time;
	int group_number=0,remote_size=0,local_size=0,tmp_buf_size=0,local_rank;
	MPI_Status status;
	int *subgroup_ranks;
	MPI_Group world_group,subgroup;
	MPI_Comm new_comm;
	int *receive_buff=NULL,*send_buff=NULL;
	//Determine which one is low group
	if(rank<n_senders){
		local_rank=rank;
		remote_size=n_receivers;
		local_size=n_senders;
	}else{
		local_rank=rank-n_senders;
		remote_size=n_senders;
		local_size=n_receivers;
	}
        start=MPI_Wtime();
	//WORLD group
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	//Data creations
	create_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,&sendcount,&recvcount,dim_x,size1,size2);
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
	if(remote_size>local_size){
		group_number=remote_size/local_size;
		tmp_buf_size=((remote_size+local_size-1)/local_size)*recvcount;
		if(remote_size%local_size){
			tmp_buf_size+=recvcount*local_size;
		}
	}else{
		group_number=local_size/remote_size;
		tmp_buf_size=recvcount;
	}
	if(tmp_buf_size>0){
		tmp_buf=malloc(sizeof(int)*tmp_buf_size);
	}
	tmp_buf_ptr=tmp_buf;
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
        start=MPI_Wtime();
	//Emulation for full-duplex comm algorithm
	if(remote_size<local_size){
		if(local_rank/remote_size<group_number){
			if(rank<n_senders){
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					n_senders+local_rank/group_number, 0,
					tmp_buf, recvcount, MPI_INT,
					n_senders+local_rank/group_number, 0,
					MPI_COMM_WORLD, &status);
			}else{
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					local_rank/group_number, 0,
					tmp_buf, recvcount, MPI_INT,
					local_rank/group_number, 0,
					MPI_COMM_WORLD, &status);
			}
		}else{
			if(rank<n_senders){
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					n_senders+local_rank%remote_size, 0,
					tmp_buf, recvcount, MPI_INT,
					n_senders+local_rank%remote_size, 0,
					MPI_COMM_WORLD, &status);
			}else{
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					local_rank%remote_size, 0,
					tmp_buf, recvcount, MPI_INT,
					local_rank%remote_size, 0,
					MPI_COMM_WORLD, &status);
			}
		}

		if(local_rank/remote_size<group_number){
			subgroup_ranks=(int*)malloc(remote_size*sizeof(int));
			for(i=0;i<remote_size;i++){
				if(rank<n_senders){
					subgroup_ranks[i]=local_rank%group_number+i*group_number;
				}else{
					subgroup_ranks[i]=n_senders+local_rank%group_number+i*group_number;
				}
			}
			MPI_Group_incl(world_group, remote_size, subgroup_ranks, &subgroup);
		}else{
			subgroup_ranks=(int*)malloc((local_size%remote_size)*sizeof(int));
			for(i=0;i<local_size%remote_size;i++){
				if(rank<n_senders){
					subgroup_ranks[i]=i+remote_size*group_number;
				}else{
					subgroup_ranks[i]=n_senders+i+remote_size*group_number;
				}
			}
			MPI_Group_incl(world_group, local_size%remote_size, subgroup_ranks, &subgroup);
		}
		MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &new_comm);
		MPI_Allgather ( 
			tmp_buf,
			recvcount,
			MPI_INT,
			recvbuf,
			recvcount,
			MPI_INT,
			new_comm);
		free(subgroup_ranks);
		MPI_Group_free(&subgroup);
		MPI_Comm_free(&new_comm);
		if(local_rank/remote_size==0&&local_rank<local_size%remote_size){
			if(rank<n_senders){
	  			MPI_Send(recvbuf+sizeof(int)*(local_size%remote_size)*recvcount, recvcount*(remote_size-(local_size%remote_size)), MPI_INT, group_number*remote_size+local_rank, 4,MPI_COMM_WORLD);
			}else{
				MPI_Send(recvbuf+sizeof(int)*(local_size%remote_size)*recvcount, recvcount*(remote_size-(local_size%remote_size)), MPI_INT, n_senders+group_number*remote_size+local_rank, 4,MPI_COMM_WORLD);
			}
		}else if(local_rank/remote_size==group_number){
			if(rank<n_senders){
				MPI_Recv(recvbuf+sizeof(int)*(local_size%remote_size)*recvcount, recvcount*(remote_size-(local_size%remote_size)), MPI_INT, local_rank%remote_size, 4, MPI_COMM_WORLD, &status);
			}else{
				MPI_Recv(recvbuf+sizeof(int)*(local_size%remote_size)*recvcount, recvcount*(remote_size-(local_size%remote_size)), MPI_INT, n_senders+local_rank%remote_size, 4, MPI_COMM_WORLD, &status);	
			}
		}

	}else{
		for(i=0;i<group_number;i++){
			if(rank<n_senders){
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					n_senders+local_rank*group_number+i, 0,
					tmp_buf_ptr, recvcount, MPI_INT,
					n_senders+local_rank*group_number+i, 0,
					MPI_COMM_WORLD, &status);
			}else{
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					local_rank*group_number+i, 0,
					tmp_buf_ptr, recvcount, MPI_INT,
					local_rank*group_number+i, 0,
					MPI_COMM_WORLD, &status);
			}
			tmp_buf_ptr+=recvcount*sizeof(int);
		}
		if(local_rank<remote_size%local_size){
			if(rank<n_senders){
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					n_senders+local_rank+group_number*local_size, 0,
					tmp_buf_ptr, recvcount, MPI_INT,
					n_senders+local_rank+group_number*local_size, 0,
					MPI_COMM_WORLD, &status);
			}else{
				MPI_Sendrecv(sendbuf, sendcount, MPI_INT,
					local_rank+group_number*local_size, 0,
					tmp_buf_ptr, recvcount, MPI_INT,
					local_rank+group_number*local_size, 0,
					MPI_COMM_WORLD, &status);				
			}
		}
		subgroup_ranks=(int*)malloc(local_size*sizeof(int));
		for(i=0;i<local_size;i++){
			if(rank<n_senders){
				subgroup_ranks[i]=i;
			}else{
				subgroup_ranks[i]=n_senders+i;
			}
		}
		MPI_Group_incl(world_group, local_size, subgroup_ranks, &subgroup);
		MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &new_comm);
	        if(remote_size%local_size){
			tmp_buf_ptr+=recvcount*sizeof(int);
			MPI_Allgather ( 
				tmp_buf,
				recvcount*group_number,
				MPI_INT,
				recvbuf,
				recvcount*group_number,
				MPI_INT,
				new_comm);
			MPI_Allgather ( 
				tmp_buf_ptr-recvcount*sizeof(int),
				recvcount,
				MPI_INT,
				tmp_buf_ptr,
				recvcount,
				MPI_INT,
				new_comm);
			if (tmp_buf_size != 0) {
				memcpy(recvbuf+sizeof(int)*group_number*recvcount*local_size,tmp_buf_ptr,sizeof(int) * recvcount * (remote_size%local_size));
			}

		}else{
			MPI_Allgather ( 
				tmp_buf,
				recvcount*group_number,
				MPI_INT,
				recvbuf,
				recvcount*group_number,
				MPI_INT,
				new_comm);
		}
		free(subgroup_ranks);
		MPI_Group_free(&subgroup);
		MPI_Comm_free(&new_comm);
	}
	if(tmp_buf_size>0){
		free(tmp_buf);
	}
	//Timing
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	//Check if receivers actually received results.
	#if DEBUG==1
	validate_result(rank,n_senders,n_receivers,receive_buff,recvcount);
	#endif
	if(rank==0){
		printf("Full-duplex Allgather emulation setup=%lf,processing=%lf\n",setup_time,allgather_time);
	}
	free(sendbuf);
	free(recvbuf);
}
/*
  Experiments with default MPI library for intergroup Allgather.
  We time the execution time of MPI_Allgather only. (assuming intergroup communicator is available at the beginning.)
    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. size1: number of dim_x elements in group A
    6. size2: number of dim_x elements in group B
*/
double bipartite_allgather_full_duplex(int rank,int dim_x,int n_senders,int n_receivers,int size1,int size2){
	//local variables
	void *sendbuf=NULL, *recvbuf=NULL;
	int color,sendcount, recvcount;
        double allgather_time=0,setup_time=0,start,total_time;
	MPI_Comm local_comm, remote_comm;
	int *receive_buff=NULL,*send_buff=NULL;
        start=MPI_Wtime();
	//WORLD group
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	/* split MPI processes into two groups */
	color = (rank < n_senders) ? 0 : 1;
	err = MPI_Comm_split(MPI_COMM_WORLD, color, rank, &local_comm); ERR
	/* obtain the inter communicator from the other group */
	err = MPI_Intercomm_create(local_comm, 0, MPI_COMM_WORLD,
                               (color == 0) ? n_senders : 0, 1, &remote_comm); ERR
	/* only the processes in 1st group send and only 2nd group receive */
	//printf("rank=%d\n",rank);
	create_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,&sendcount,&recvcount,dim_x,size1,size2);
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
        start=MPI_Wtime();
	/* use Allgather to achieve effect of all Bcast from the 1st group * to 2nd group */
	err = MPI_Allgather(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, remote_comm); ERR\
	//Timing
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	double result;
	MPI_Reduce(&allgather_time, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//Check if receivers actually received results.
	#if DEBUG==1
	validate_result(rank,n_senders,n_receivers,receive_buff,recvcount);
	#endif
	MPI_Comm_free(&remote_comm);
	MPI_Comm_free(&local_comm);
	if(rank==0){
		printf("Full-duplex Allgather Benchmark setup=%lf,processing=%lf\n",setup_time,allgather_time);
	}
	free(sendbuf);
	free(recvbuf);
	return result;
}
/*
  Experiments with emulation of the root gathering algorithm using MPI_Gather and MPI_Bcast.
    1. rank: rank of process
    2. dim_x: data size unit
    3. n_senders: size of group A
    4. n_receivers: size of group B
    5. size1: number of dim_x elements in group A
    6. size2: number of dim_x elements in group B
*/
double bipartite_allgather_full_duplex_benchmark_emulation(int rank,int dim_x,int n_senders,int n_receivers,int size1,int size2){
	//local variables
	void *sendbuf=NULL, *recvbuf=NULL,*data=NULL;
	int sendcount, recvcount,local_size,remote_size,i;
        double allgather_time=0,setup_time=0,start,total_time;
	int* subgroup_ranks;
	int *receive_buff=NULL,*send_buff=NULL;
        start=MPI_Wtime();
	//WORLD group
	MPI_Group world_group;
	MPI_Group left_broadcast_group;
	MPI_Comm left_broadcast_comm;
	MPI_Group right_broadcast_group;
	MPI_Comm right_broadcast_comm;
	MPI_Group gather_group;
	MPI_Comm gather_comm;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	/* split MPI processes into two groups */
	if(rank>=n_senders){
		local_size=n_receivers;
		subgroup_ranks=(int*)malloc(n_receivers*sizeof(int));
		for(i=0;i<n_receivers;i++){
			subgroup_ranks[i]=n_senders+i;
		}
	}else{
		local_size=n_senders;
		subgroup_ranks=(int*)malloc(n_senders*sizeof(int));
		for(i=0;i<n_senders;i++){
			subgroup_ranks[i]=i;
		}
	}
	MPI_Group_incl(world_group, local_size, subgroup_ranks, &gather_group);
	MPI_Comm_create_group(MPI_COMM_WORLD, gather_group, 0, &gather_comm);
	free(subgroup_ranks);
	/* obtain the inter communicator from the other group */
	if(rank==0||rank>=n_senders){
		local_size=n_receivers+1;
		subgroup_ranks=(int*)malloc(local_size*sizeof(int));
		subgroup_ranks[0]=0;
		for(i=0;i<n_receivers;i++){
			subgroup_ranks[i+1]=n_senders+i;
		}
		//printf("right group %d\n",rank);
		MPI_Group_incl(world_group, local_size, subgroup_ranks, &right_broadcast_group);
		MPI_Comm_create_group(MPI_COMM_WORLD, right_broadcast_group, 0, &right_broadcast_comm);
		free(subgroup_ranks);
	}
	if(rank<=n_senders){
		local_size=n_senders+1;
		subgroup_ranks=(int*)malloc(local_size*sizeof(int));
		subgroup_ranks[0]=n_senders;
		for(i=0;i<n_senders;i++){
			subgroup_ranks[i+1]=i;
		}
		//printf("left group %d\n",rank);
		MPI_Group_incl(world_group, local_size, subgroup_ranks, &left_broadcast_group);
		MPI_Comm_create_group(MPI_COMM_WORLD, left_broadcast_group, 0, &left_broadcast_comm);
		free(subgroup_ranks);
	}
	create_data(rank, n_senders,n_receivers,&send_buff,&receive_buff,&sendcount,&recvcount,dim_x,size1,size2);
	recvbuf=(void*)receive_buff;
	sendbuf=(void*)send_buff;
	if(rank<n_senders){
		local_size=n_senders;
		remote_size=n_receivers;
	}else{
		remote_size=n_senders;
		local_size=n_receivers;
	}
	MPI_Barrier(MPI_COMM_WORLD);
        total_time=MPI_Wtime();
	/* use Allgather to achieve effect of all Bcast from the 1st group * to 2nd group */
	if(rank==0){
		data=(int*)malloc(local_size*size1*dim_x*sizeof(int));
	}else if(rank==n_senders){
		data=(int*)malloc(local_size*size2*dim_x*sizeof(int));
	}
        start=MPI_Wtime();
	if(rank<n_senders){
		MPI_Gather(sendbuf, dim_x*size1, MPI_INT, data, dim_x*size1, MPI_INT,0, gather_comm);
	}else{
		MPI_Gather(sendbuf, dim_x*size2, MPI_INT, data, dim_x*size2, MPI_INT,0, gather_comm);
	}
	if(rank<n_senders){
		if(rank==0){
			MPI_Bcast( data, dim_x*local_size*size1, MPI_INT, 0, right_broadcast_comm);
			//printf("broadcast to right\n");
		}
		MPI_Bcast( recvbuf, dim_x*remote_size*size2, MPI_INT, 0, left_broadcast_comm);
	}else{
		MPI_Bcast( recvbuf, dim_x*remote_size*size1, MPI_INT, 0, right_broadcast_comm);
		if(rank==n_senders){
			MPI_Bcast( data, dim_x*local_size*size2, MPI_INT, 0, left_broadcast_comm);
			//printf("broadcast to left\n");
		}
	}
	//Timing
        total_time=MPI_Wtime();
	allgather_time=total_time-start;
	double result;
	MPI_Reduce(&allgather_time, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//Check if receivers actually received results.
	//printf("rank=%d\n",rank);
	#if DEBUG==1
	validate_result(rank,n_senders,n_receivers,receive_buff,recvcount);
	#endif
	if(rank==0){
		printf("Full-duplex Allgather Benchmark setup=%lf,processing=%lf\n",setup_time,allgather_time);
	}
	//printf("rank=%d\n",rank);
	
	if(rank==0||rank>=n_senders){
		MPI_Group_free(&right_broadcast_group);
		MPI_Comm_free(&right_broadcast_comm);
	}
	if(rank<=n_senders){
		MPI_Group_free(&left_broadcast_group);
		MPI_Comm_free(&left_broadcast_comm);
	}
	MPI_Group_free(&gather_group);
	MPI_Comm_free(&gather_comm);
	free(sendbuf);
	free(recvbuf);
	return result;
}

int main(int argc, char **argv){
	int procs,rank,i,size1,size2,dim_x,n_senders,n_receivers,pattern=0,method=0,iterations=10;
	double sum_time,min,max,temp;
	double *temp_timings;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&procs);
	/* command-line arguments */
	while ((i = getopt(argc, argv, "hp:a:b:s:r:d:t:p:n:")) != EOF){
		switch(i) {
			case 'a': size1 = atoi(optarg);
				break;
			case 'b': size2 = atoi(optarg);
				break;
			case 's': n_senders = atoi(optarg);
				break;
			case 'r': n_receivers = atoi(optarg);
				break;
			case 'd': dim_x = atoi(optarg);
				break;
			case 't': method = atoi(optarg);
				break;
			case 'p': method = atoi(optarg);
				break;
			case 'n': iterations = atoi(optarg);
				break;
			case 'h':
				default:  if (rank==0) usage(argv[0]);
					MPI_Finalize();
			return 1;
		}
	}
	if(rank==0){
		printf("n_senders=%d,n_receivers=%d,dim_x=%d,size1=%d,size2=%d,method=%d,pattern=%d\n",n_senders,n_receivers,dim_x,size1,size2,method,pattern);
	}
	if(procs!=n_senders+n_receivers){
		if(rank==0){
			printf("Number of processors should be the sum of number of receivers and number of senders.\n");
		}
		MPI_Finalize();
		return 0;
	}
	if(method==4){
		int v_size1[n_senders];
		int v_size2[n_receivers];
		for(i=0;i<n_senders;i++){
			if(pattern==0){
				v_size1[i]=size1;
			}else{
				v_size1[i]=2*(i+1)*((size1+n_senders-1)/n_senders);
			}
		}
		for(i=0;i<n_receivers;i++){
			if(pattern==0){
				v_size2[i]=size2;
			}else{
				v_size2[i]=2*(i+1)*((size2+n_receivers-1)/n_receivers);
			}
		}
		sum_time=0;
		min=-1;
		max=-1;
		for(i=0;i<iterations;i++){
			temp_timings=bipartite_allgatherv_universal_full_duplex_emulation(rank,dim_x,n_senders,n_receivers,v_size1, v_size2);
			temp=temp_timings[0]+temp_timings[1]+temp_timings[2];
			if(min==-1||min>temp){
				min=temp;
			}
			if(max==-1||max<temp){
				max=temp;
			}
			sum_time+=temp;
		}
		if(rank==0){
			printf("Time used for universal full duplex allgatherv=%lf, removing extreme=%lf\n",sum_time,sum_time-min-max);
		}
	}else if(method==3){
		int v_size1[n_senders];
		int v_size2[n_receivers];
		for(i=0;i<n_senders;i++){
			if(pattern==0){
				v_size1[i]=size1;
			}else{
				v_size1[i]=2*(i+1)*((size1+n_senders-1)/n_senders);
			}
		}
		for(i=0;i<n_receivers;i++){
			if(pattern==0){
				v_size2[i]=size2;
			}else{
				v_size2[i]=2*(i+1)*((size2+n_receivers-1)/n_receivers);
			}
		}
		sum_time=0;
		min=-1;
		max=-1;
		for(i=0;i<iterations;i++){
			temp=bipartite_allgatherv_full_duplex(rank,dim_x,n_senders,n_receivers,v_size1, v_size2);
			if(min==-1||min>temp){
				min=temp;
			}
			if(max==-1||max<temp){
				max=temp;
			}
			sum_time+=temp;
		}
		if(rank==0){
			printf("Time used for benchmark full duplex allgatherv=%lf, removing extreme=%lf\n",sum_time,sum_time-min-max);
		}
	}else if(method==2){
		sum_time=0;
		min=-1;
		max=-1;
		for(i=0;i<iterations;i++){
			temp=bipartite_allgather_simple_universal_full_duplex_emulation(rank,dim_x,n_senders,n_receivers,atoi(argv[4]),atoi(argv[5]));
			if(min==-1||min>temp){
				min=temp;
			}
			if(max==-1||max<temp){
				max=temp;
			}
			sum_time+=temp;
		}
		if(rank==0){
			printf("Time used for simple universal full duplex allgather=%lf, removing extreme=%lf\n",sum_time,sum_time-min-max);
		}
	}else if(method==1){
		sum_time=0;
		min=-1;
		max=-1;
		for(i=0;i<iterations;i++){
			temp=bipartite_allgather_universal_full_duplex_emulation(rank,dim_x,n_senders,n_receivers,atoi(argv[4]),atoi(argv[5]));
			if(min==-1||min>temp){
				min=temp;
			}
			if(max==-1||max<temp){
				max=temp;
			}
			sum_time+=temp;
		}
		if(rank==0){
			printf("Time used for universal full duplex allgather=%lf, removing extreme=%lf\n",sum_time,sum_time-min-max);
		}
	}else if(method==0){
		sum_time=0;
		min=-1;
		max=-1;
		for(i=0;i<iterations;i++){
			temp=bipartite_allgather_full_duplex(rank,dim_x,n_senders,n_receivers,atoi(argv[4]),atoi(argv[5]));
			if(min==-1||min>temp){
				min=temp;
			}
			if(max==-1||max<temp){
				max=temp;
			}
			sum_time+=temp;
		}
		if(rank==0){
			printf("Time used for benchmark allgather=%lf, removing extreme=%lf\n",sum_time,sum_time-min-max);
		}
	}
	MPI_Finalize();
	return 0;
}
