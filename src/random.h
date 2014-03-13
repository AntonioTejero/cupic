/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef GSLRAND_H
#define GSLRAND_H

/****************************** HEADERS ******************************/

#include <gsl/gsl_rng.h>        //gsl library for random number generation
#include <gsl/gsl_randist.h>    //gsl library for random number generation (distribution functions)
#include <curand_kernel.h>      //curand library for random number generation (__device__ functions)

/************************ SIMBOLIC CONSTANTS *************************/

#define CURAND_BLOCK_DIM 64      //block dimension for curand kernels

/************************ FUNCTION PROTOTIPES ************************/

#endif
