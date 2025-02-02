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
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define DIAGNOSTIC_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function

int number_of_particles(int *d_bm);
void particles_snapshot(particle *d_p, int * d_bm, string filename, double t);
void mesh_snapshot(double *d_m, string filename);
void save_bm(int * d_bm, string filename);
void save_bins(int * d_bm, particle *d_p, string filename);

// device kernels



// device functions 



#endif