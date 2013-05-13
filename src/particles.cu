/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "particles.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void particle_mover(int ncx, int ncy, particle *elec, unsigned int *e_bm, particle *ions, unsigned int *i_bm, double *Ex, double *Ey) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  double *Fx, *Fy;        // force suffered for each particle (electrons or ions)
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel
  griddim = ncy-1;
  blockdim = CHARGE_DEP_BLOCK_DIM;
  
  // allocate host memory for electron forces
  Fx = new double[e_bm[ncy-2]];
  
  
  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void leap_frog_step(double dt, particle *p, double *Fx, double *Fy) 
{
  
  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/

