/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef CUDA_H
#define CUDA_H

/****************************** HEADERS ******************************/

#include "stdh.h"

/************************ SIMBOLIC CONSTANTS *************************/



/************************ FUNCTION PROTOTIPES ************************/

// host function
void cu_check(cudaError_t cuError, const string file, const int line);
void cu_sync_check(const string file, const int line);

// device kernels



// device functions 



#endif