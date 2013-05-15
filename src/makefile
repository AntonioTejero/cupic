# Configuration

CC = g++
NVCC = nvcc
NVCCFLAGS = -arch=sm_21
GSLFLAGS = -lgsl -lgslcblas


# Makefile orders

CUPIC : main.o init.o cc.o mesh.o particles.o
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -lcuda -lcudart main.o init.o cc.o -o cupic
	rm *~ -f
	mv ./cupic ../bin/cupic

main.o : main.cu 
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -dc main.cu -o main.o 

init.o : init.cu
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -dc init.cu -o init.o 

cc.o : cc.cu
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -dc cc.cu -o cc.o
	
mesh.o : mesh.cu
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -dc mesh.cu -o mesh.o
	
particles.o : particles.cu
	$(NVCC) $(NVCCFLAGS) $(GSLFLAGS) -dc particles.cu -o particles.o
	
.PHONY : clean open

clean :
	rm *.o *~ *.dat -f
	clear

open :
	kate stdh.h main.cu init.h init.cu cc.h cc.cu mesh.h mesh.cu particles.h particles.cu makefile &