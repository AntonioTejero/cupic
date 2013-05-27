/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef INIT_H
#define INIT_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "gslrand.h"
#include "mesh.h"
#include "particles.h"


/************************ FUNCTION PROTOTIPES ************************/

// host functions
void initialize (double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_e_bm, unsigned int **d_i_bm);

void read_input_file (double *qi, double *qe, double *mi, double *me, double *kti, double *kte, double *phi_p, double *n, double *Lx, double *Ly, double *ds, double *dt, double *epsilon0);

double init_qi(void);
double init_qe(void);
double init_mi(void);
double init_me(void);
double init_kti(void);
double init_kte(void);
double init_phi_p(void);
double init_n(void);
double init_Lx(void);
double init_Ly(void);
double init_ds(void);
double init_dt(void);
double init_dtin_i(void);
double init_dtin_e(void);
double init_epsilon0(void);
int init_ncx(void);
int init_ncy(void);
int init_nnx(void);
int init_nny(void);

// device kernels
__global__ void fix_velocity(double dt, double m, particle *g_p, unsigned int *g_bm, double *g_Fx, double *g_Fy);

#endif