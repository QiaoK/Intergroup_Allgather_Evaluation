## Intergroup Allgather and Allgatherv Implemented Using MPI functions

The program [./allgather_test](intergroup_allgather.c) runs the intergroup all-to-all broadcast in
a loop. The number of iterations can be set
by the command-line option `-n`. The default is 10.

* Compile command:
  * Edit file (Makefile to customize the C compiler and compile
    options.
  * Run command `make` to compile and generate the executable program named
    `allgather_test`.

* Run command:
  * Command-line options:
    ```
      % ./allgather_test -h
	Usage: ./allgather_test [OPTION]... [FILE]...
	       [-h] Print help
	       [-s] number of MPI processes in group A
	       [-r] number of MPI processes in group B
	       [-a] message count per process in group A
	       [-b] message count per process in group B
	       [-d] message block unit size
	       [-n] number of trials for the experiments
	       [-p] only significant for Allgatherv (when t = 3, 4)
		   0: regular message distribution
		   1: irregular message distribution
	       [-t] which evaluation
		   0: Intergroup Allgather with native MPI library
		   1: Intergroup Allgather with algorithm 2 of EuroMPI paper
		   2: Intergroup Allgather with algorithm 1 of PARCO paper
		   3: Intergroup Allgatherv with native MPI library
		   4: Intergroup Allgather with algorithm 2 of PARCO paper
    ```
* Example outputs on screen
  * Suppose we have 10 processes in group A and 22 processes in group B. Every process in group A sends 15 data blocks to group B and every process in group B sends 20 data blocks to group A. Every data block has 11 integers. We want to evaluate Algorithm 2 proposed in PARCO paper with 5 iterations.
  ```
    % mpiexec -n 32 ./allgather_test -a 15 -b 20 -s 10 -r 22 -d 11 -p 0 -t 4 -n 5
	n_senders=10,n_receivers=22,dim_x=11,size1=10,size2=20,method=4,pattern=0
	Universal full duplex allgatherv time=0.059017(message size time)+0.037100(intergroup message transfer)+0.066018(intragroup allgather)=0.162135
	Universal full duplex allgatherv time=0.036167(message size time)+0.020993(intergroup message transfer)+0.039993(intragroup allgather)=0.097153
	Universal full duplex allgatherv time=0.041959(message size time)+0.021000(intergroup message transfer)+0.039992(intragroup allgather)=0.102951
	Universal full duplex allgatherv time=0.042807(message size time)+0.021071(intergroup message transfer)+0.054091(intragroup allgather)=0.117969
	Universal full duplex allgatherv time=0.050991(message size time)+0.035053(intergroup message transfer)+0.054941(intragroup allgather)=0.140985
	Time used for universal full duplex allgatherv=0.621193, removing extreme=0.361905
  ```

## Questions/Comments:
email: qiao.kang@eecs.northwestern.edu

Copyright (C) 2019, Northwestern University.

See [COPYRIGHT](COPYRIGHT) notice in top-level directory.

