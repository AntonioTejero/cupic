/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef PARTICLES_H
#define PARTICLES_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "diagnostic.h"
#include "dynamic_sh_mem.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define PAR_MOV_BLOCK_DIM 512   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void particle_mover(particle *d_e, int *d_e_bm, particle *d_i, int *d_i_bm, double *Ex, double *Ey);

// device kernels
__global__ void fast_grid_to_particle(int nnx, int q, double ds, particle *g_p, int *g_bm, double *g_Ex, double *g_Ey, double *g_Fx, double *g_Fy);
__global__ void leap_frog_step(double dt, double m, particle *g_p, int *g_bm, double *g_Fx, double *g_Fy);

// device functions 


#endif