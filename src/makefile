# Configuration

CC = g++
NVCC = nvcc
NVCCFLAGS = -arch=sm_21
GSLFLAGS = -lgsl -lgslcblas

CUPIC : main.cu init.cu cc.cu
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) main.cu init.cu cc.cu mesh.cu particles.cu -o cupic
	mv ./cupic ../bin/cupic
	rm *~ -f

# CUPIC : main.o init.o cc.o
# 	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) main.o init.o cc.o -o cupic
# 	rm *.o *~ -f
# 	mv ./oolppic ../bin/oolppic
# 
# main.o : main.cu 
# 	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -c main.cu -o main.o 
# 
# init.o : init.cu
# 	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -c init.cu -o init.o 
# 
# cc.o : cc.cu
# 	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -c cc.cu -o cc.o
	
.PHONY : clean open

clean :
	rm *.o *~ *.dat -f

open :
	kate stdh.h main.cu init.h init.cu cc.h cc.cu mesh.h mesh.cu particles.h particles.cu makefile &