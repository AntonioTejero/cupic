# Configuration

CC = g++
NVCC = nvcc
NVCCFLAGS = -arch=sm_21
LINKERFLAGS = -lgsl -lgslcblas -lcuda -lcudart

OBJECTS = main.o init.o cc.o mesh.o particles.o


# Makefile orders

CUPIC : $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LINKERFLAGS) $(OBJECTS) -o cupic
	rm *~ -f
	mv ./cupic ../bin/cupic

main.o : main.cu 
	$(NVCC) $(NVCCFLAGS) -dc main.cu -o main.o 

init.o : init.cu
	$(NVCC) $(NVCCFLAGS) -dc init.cu -o init.o 

cc.o : cc.cu
	$(NVCC) $(NVCCFLAGS) -dc cc.cu -o cc.o
	
mesh.o : mesh.cu
	$(NVCC) $(NVCCFLAGS) -dc mesh.cu -o mesh.o
	
particles.o : particles.cu
	$(NVCC) $(NVCCFLAGS) -dc particles.cu -o particles.o
	
.PHONY : clean open lines

clean :
	rm *.o *~ *.dat -f
	clear

open :
	kate stdh.h main.cu init.h init.cu cc.h cc.cu mesh.h mesh.cu particles.h particles.cu makefile &
	
lines :
	git ls-files | xargs wc -l