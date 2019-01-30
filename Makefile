CC=mpicc
CFLAGS= -Wall -Wextra -O2
LIBS = -lm
ALLGATHER_TEST_OBJS = intergroup_allgather.c
allgather_test : $(ALLGATHER_TEST_OBJS)
	$(CC) -o $@ $(ALLGATHER_TEST_OBJS) $(LIBS)
%.o: %.c
	$(CC) $(CFLAGS) -c $<  
clean:
	rm -rf *.o
	rm -rf allgather_test
