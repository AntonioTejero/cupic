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

/************************ SIMBOLIC CONSTANTS *************************/

#define PARTICLE_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function


// device kernels


// device functions 


// variable for allowing dynamic allocation of __shared__ memory (used in charge_deposition, poisson_solver, )
extern __shared__ double sh_mem[];


#endif