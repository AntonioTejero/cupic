/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef DIAGNOSTIC_H
#define DIAGNOSTIC_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define DIAGNOSTIC_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function

unsigned int number_of_particles(unsigned int *d_bm);
void snapshot(particle *d_p, unsigned int * d_bm, string filename);

// device kernels



// device functions 



#endif