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
#include "init.h"
#include "dynamic_sh_mem.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define CHARGE_DEP_BLOCK_DIM 512   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void charge_deposition(double *d_rho, particle *d_e, int *d_e_bm, particle *d_i, int *d_i_bm);
void poisson_solver(double max_error, double *d_rho, double *d_phi);
void field_solver(double *d_phi, double *d_Ex, double *d_Ey);

// device kernels
__global__ void fast_particle_to_grid(int nnx, double ds, double *rho, particle *elec, int *e_bm, particle *ions, int *i_bm);
__global__ void jacobi_iteration (dim3 blockdim, double ds, double epsilon0, double *rho, double *phi, double *block_error);
__global__ void field_derivation (double ds, double *phi_global, double *Ex_global, double *Ey_global);

// device functions (overload atomic functions for double precision support)
__device__ double atomicAdd(double* address, double val);
__device__ double atomicSub(double* address, double val);

#endif
