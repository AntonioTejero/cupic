/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef CC_H
#define CC_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "gslrand.h"
#include "diagnostic.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define BINING_BLOCK_DIM 512   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void cc (double t, int *d_e_bm, particle **d_e, int *d_i_bm, particle **d_i, double *d_Ex, double *d_Ey);
void particle_bining(double Lx, double dy, int ncy, int *bm, int *new_bm, particle *p);


// device kernels
__global__ void particle_defragmentation(double Lx, double dy, int *bm, int *new_bm, particle *p);
__global__ void particle_rebracketing(int *bm, int *new_bm, particle *p);

#endif 
