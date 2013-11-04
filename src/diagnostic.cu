/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "diagnostic.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

int number_of_particles(int *d_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int ncy = init_ncy();      // number of cells in y dimension
  
  cudaError_t cuError;
  int h_bm[2*ncy];
  int ini, fin;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cuError = cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);

  // evaluate number of particles
  ini = 0;
  fin = 2*ncy-1;

  while (h_bm[fin] < 0 && fin > 0) fin -= 2;
  while (h_bm[ini] < 0 && ini < fin) ini += 2;
  
  if (ini > fin) return 0;
  else return h_bm[fin]-h_bm[ini]+1;
}


/**********************************************************/

void particles_snapshot(particle *d_p, int * d_bm, string filename) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory  
  particle *h_p;
  int N;
  FILE *file;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // evaluate number of particles in the vector
  N = number_of_particles(d_bm);
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(N*sizeof(particle));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_p, d_p, N*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save snapshot to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file = fopen(filename.c_str(), "w");
  
  for (int i = 0; i < N; i++) 
  {
    fprintf(file, " %.17e %.17e %.17e %.17e \n", h_p[i].x, h_p[i].y, h_p[i].vx, h_p[i].vy);
  }
  
  fclose(file);
  
  // free host memory
  free(h_p);
  
  return;
}

/**********************************************************/

void mesh_snapshot(double *d_m, string filename) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory 
  static const int nnx = init_nnx();
  static const int nny = init_nny();
  double *h_m;
  FILE *file;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for mesh vector
  h_m = (double *) malloc(nnx*nny*sizeof(double));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_m, d_m, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save snapshot to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file = fopen(filename.c_str(), "w");
  
  for (int i = 0; i < nnx; i++) 
  {
    for (int j = 0; j < nny; j++) 
    {
      fprintf(file, " %d %d %.17e \n", i, j, h_m[i+j*nnx]);
    }
    fprintf(file, "\n");
  }
  
  fclose(file);
  
  // free host memory
  free(h_m);
  
  return;
}

/**********************************************************/

void save_bm(int * d_bm, string filename)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int ncy = init_ncy();      // number of cells in y dimension
  int h_bm[2*ncy];
  FILE *file;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cuError = cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);

  // save bookmarks to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file = fopen(filename.c_str(), "w");

  for (int i = 0; i<2*ncy; i+=2)
  {
    fprintf(file, " %d %d %d \n", i, h_bm[i], h_bm[i+1]);
  }

  fclose(file);
  
  return;
}

/**********************************************************/

void save_bins(int *d_bm, particle *d_p, string filename)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();      // spacial step
  particle *h_p;
  int N;
  FILE *file;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // evaluate number of particles in the vector
  N = number_of_particles(d_bm);
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(N*sizeof(particle));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_p, d_p, N*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save bins to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file = fopen(filename.c_str(), "w");
  
  for (int i = 0; i < N; i++) 
  {
    fprintf(file, " %d %d \n", i, int(h_p[i].y/ds));
  }
  
  fclose(file);

  //free host memory for particle vector
  free(h_p);
  
  return;
}

/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/
