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

/************************ SIMBOLIC CONSTANTS *************************/

#define BINING_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void particle_bining(double dy, int ncy, int *bookmark, int *new_bookmark, particle **p);
// void cc_abs_inj (double t, double dt int *e_bookmark, int *e_new_bookmark, particle **e, int *i_bookmark, int * i_new_bookmark, particle **i);


// device kernels
__global__ void particle_defragmentation(int *bookmark, int *new_bookmark, double dy, particle *p);
__global__ void particle_rebracketing(int *bookmark, int *new_bookmark, particle *p);

#endif