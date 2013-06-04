/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "diagnostic.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

unsigned int number_of_particles(unsigned int *d_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int ncy = init_ncy();      // number of cells in y dimension
  
  unsigned int num_particles;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cudaMemcpy (&num_particles, d_bm+2*ncy-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  num_particles += 1;
  
  return num_particles;
}


/**********************************************************/



/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/