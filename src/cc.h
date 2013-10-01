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
#include "dynamic_sh_mem.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define BINING_BLOCK_DIM 256   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void cc (double t, int *d_e_bm, particle **d_e, int *d_i_bm, particle **d_i, double *d_Ex, double *d_Ey);
void particle_cc(double t, double *tin, double dtin, double kt, double m, int *d_bm, particle **d_p, double *d_Ex, double *d_Ey);
void particle_bining(double Lx, double dy, int ncy, int *bm, int *new_bm, particle *p);
void abs_emi_cc(double t, double *tin, double dtin, double kt, double m, int *d_bm, int *d_new_bm, particle **d_p, double *d_Ex, double *d_Ey);
void cyclic_cc(int ncy, double Lx, int *d_bm, particle *d_p);


// device kernels
__global__ void pDefragDown(double ds, int *g_new_bm, particle *g_p);
__global__ void pDefragUp(double ds, int *g_new_bm, particle *g_p);
__global__ void pRebracketing(int *bm, int *new_bm, particle *p, int *n);
__global__ void bmHandler(int *bm, int *n, int ncy);
__global__ void pCyclicCC(double Lx, int *g_bm, particle *g_p);

#endif 
