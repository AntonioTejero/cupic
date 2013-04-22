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

#ifndef RAND_H
#define RAND_H

#include <gsl/gsl_rng.h>        //gsl library for random number generation
#include <gsl/gsl_randist.h>    //gsl library for random number generation

#endif

/************************ FUNCTION PROTOTIPES ************************/

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, unsigned int **h_bookmarke, unsigned int **h_bookmarki, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_bookmarke, unsigned int **d_bookmarki);

void read_input_file (double *h_qi, double *h_qe, double *h_mi, double *h_me, double *h_kti, double *h_kte, double *h_phi_p, double *h_n, double *h_Lx, double *h_Ly, double *h_dx, double *h_dy, double *h_dz, double *h_t, double *h_dt, double *h_epsilon);

#endif