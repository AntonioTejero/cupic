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

/************************ SIMBOLIC CONSTANTS *************************/

#define BINING_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void cc (double t, unsigned int *d_e_bookmark, particle **e, unsigned int *d_i_bookmark, particle **i, double *d_Ex, double *d_Ey);
inline void particle_bining(double Lx, double dy, int ncy, unsigned int *bookmark, unsigned int *new_bookmark, particle *p);


// device kernels
__global__ void particle_defragmentation(double Lx, double dy, unsigned int *bookmark, unsigned int *new_bookmark, particle *p);
__global__ void particle_rebracketing(unsigned int *bookmark, unsigned int *new_bookmark, particle *p);

#endif