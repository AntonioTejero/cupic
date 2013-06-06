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
  
  unsigned int h_bm[2*ncy];
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  return h_bm[2*ncy-1]-h_bm[0]+1;
}


/**********************************************************/



/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/