/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef MESH_H
#define MESH_H

/****************************** HEADERS ******************************/

#include "stdh.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define CHARGE_DEP_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void fast_particle_to_grid(int ncx, int ncy, double dx, double dy, double *rho, particle *elec, unsigned int *e_bm, particle *ions, unsigned int *i_bm);

// device kernels
__global__ void charge_deposition(int ncx, int ncy, double dx, double dy, double *rho, particle *elec, unsigned int *e_bm, particle *ions, unsigned int *i_bm);

// device functions (overload atomic functions for double precision support)
__device__ double atomicAdd(double* address, double val);
__device__ double atomicSub(double* address, double val);

// variable for allowing dynamic allocation of __shared__ memory (used in charge_deposition, poisson_solver, )
extern __shared__ double sh_mem[];


#endif