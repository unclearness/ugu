#include "ugu/cuda/sparse.h"

#include <assert.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "cusparse.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "ugu/log.h"
#include "ugu/timer.h"

namespace {
// https://stackoverflow.com/questions/57334742/convert-eigensparsematrix-to-cusparse-and-vice-versa
void EigenSparseToCuSparseTranspose(const Eigen::SparseMatrix<double> &mat,
                                    int *row, int *col, double *val) {
  const int num_non0 = mat.nonZeros();
  const int num_outer = mat.cols() + 1;

  checkCudaErrors(cudaMalloc((void **)&row, sizeof(int) * num_outer));
  checkCudaErrors(cudaMalloc((void **)&col, sizeof(int) * num_non0));
  checkCudaErrors(cudaMalloc((void **)&val, sizeof(double) * num_non0));

  cudaMemcpy(row, mat.outerIndexPtr(), sizeof(int) * num_outer,
             cudaMemcpyHostToDevice);

  // for (int i = 0; i < num_outer; i++) {
  //   std::cout << mat.outerIndexPtr()[i] << " ";
  // }

  cudaMemcpy(col, mat.innerIndexPtr(), sizeof(int) * num_non0,
             cudaMemcpyHostToDevice);

  cudaMemcpy(val, mat.valuePtr(), sizeof(double) * num_non0,
             cudaMemcpyHostToDevice);
}
void EigenSparseToCsr(const Eigen::SparseMatrix<double> &mat,
                      std::vector<int> &row, std::vector<int> &col,
                      std::vector<double> &val) {
  const int num_non0 = mat.nonZeros();
  const int num_outer = mat.cols() + 1;

  row.resize(num_outer);
  std::memcpy(row.data(), mat.outerIndexPtr(), sizeof(int) * num_outer);

  col.resize(num_non0);
  std::memcpy(col.data(), mat.innerIndexPtr(), sizeof(int) * num_non0);

  val.resize(num_non0);
  std::memcpy(val.data(), mat.valuePtr(), sizeof(double) * num_non0);
}

bool SolveSparseCg(int rowsA, int colsA, int nnzA, const float *h_csrValA,
                   const int *h_csrRowPtrA, const int *h_csrColIndA,
                   const float *h_b, float *h_x, int out_col, int devID) {
  ugu::Timer<> timer;
  timer.Start();

  int nz = nnzA;
  int M = rowsA;
  int N = colsA;

  const int max_iter = 1000;
  // int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
  int *d_col, *d_row;
  int qatest = 0;
  const float tol = 1e-12f;

  // float *x, *rhs;
  float r0, r1, alpha, beta;
  float *d_val, *d_x;
  float *d_zm1, *d_zm2, *d_rm2;
  float *d_r, *d_p, *d_omega, *d_y;
  // float *val = NULL;
  float *d_valsILU0;
  float rsum, diff, err = 0.0;
  float qaerr1, qaerr2 = 0.0;
  float dot, numerator, denominator, nalpha;
  const float floatone = 1.0;
  const float floatzero = 0.0;

  int nErrors = 0;

  printf("conjugateGradientPrecond starting...\n");

  /* QA testing mode */
  // if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
  //   qatest = 1;
  // }

  /* This will pick the best possible CUDA capable device */
  cudaDeviceProp deviceProp;
  // int devID = findCudaDevice(argc, (const char **)argv);
  // printf("GPU selected Device ID = %d \n", devID);

  // if (devID < 0) {
  //   printf("Invalid GPU device %d selected,  exiting...\n", devID);
  //  exit(EXIT_SUCCESS);
  //}

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  /* Statistics about the GPU device */
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

#if 0
  /* Generate a Laplace matrix in CSR (Compressed Sparse Row) format */
  //M = N = 16384;
  //nz = 5 * N - 4 * (int)sqrt((double)N);
  I = (int *)malloc(sizeof(int) * (N + 1));   // csr row pointers for matrix A
  J = (int *)malloc(sizeof(int) * nz);        // csr column indices for matrix A
  val = (float *)malloc(sizeof(float) * nz);  // csr values for matrix A
  x = (float *)malloc(sizeof(float) * N);
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 0.0;  // Initialize RHS
    x[i] = 0.0;    // Initial solution approximation
  }
#endif

  for (int i = 0; i < rowsA; i++) {
    h_x[i] = 0.0;  // Initial solution approximation
  }

  // genLaplace(I, J, val, M, N, nz, rhs);

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  checkCudaErrors(cublasCreate(&cublasHandle));

  /* Create CUSPARSE context */
  cusparseHandle_t cusparseHandle = NULL;
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  /* Description of the A matrix */
  cusparseMatDescr_t descr = 0;
  checkCudaErrors(cusparseCreateMatDescr(&descr));
  checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  /* Allocate required memory */
  checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zm1, (N) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zm2, (N) * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_rm2, (N) * sizeof(float)));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseDnVecDescr_t vecp = NULL, vecX = NULL, vecY = NULL, vecR = NULL,
                       vecZM1 = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
  checkCudaErrors(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
  checkCudaErrors(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F));
  checkCudaErrors(cusparseCreateDnVec(&vecR, N, d_r, CUDA_R_32F));
  checkCudaErrors(cusparseCreateDnVec(&vecZM1, N, d_zm1, CUDA_R_32F));
  cusparseDnVecDescr_t vecomega = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecomega, N, d_omega, CUDA_R_32F));

  /* Initialize problem data */
  checkCudaErrors(cudaMemcpy(d_col, h_csrColIndA, nz * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_row, h_csrRowPtrA, (N + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_val, h_csrValA, nz * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_val, h_csrValA, nz * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_r, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  cusparseSpMatDescr_t matA = NULL;
  cusparseSpMatDescr_t matM_lower, matM_upper;
  cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
  cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

  checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  /* Copy A data to ILU(0) vals as input*/
  checkCudaErrors(cudaMemcpy(d_valsILU0, d_val, nz * sizeof(float),
                             cudaMemcpyDeviceToDevice));

  // Lower Part
  checkCudaErrors(cusparseCreateCsr(
      &matM_lower, N, N, nz, d_row, d_col, d_valsILU0, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  checkCudaErrors(cusparseSpMatSetAttribute(
      matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)));
  checkCudaErrors(cusparseSpMatSetAttribute(
      matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit)));
  // M_upper
  checkCudaErrors(cusparseCreateCsr(
      &matM_upper, N, N, nz, d_row, d_col, d_valsILU0, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  checkCudaErrors(cusparseSpMatSetAttribute(
      matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper)));
  checkCudaErrors(
      cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_DIAG_TYPE,
                                &diag_non_unit, sizeof(diag_non_unit)));

  /* Create ILU(0) info object */
  int bufferSizeLU = 0;
  size_t bufferSizeMV, bufferSizeL, bufferSizeU;
  void *d_bufferLU, *d_bufferMV, *d_bufferL, *d_bufferU;
  cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
  cusparseMatDescr_t matLU;
  csrilu02Info_t infoILU = NULL;

  checkCudaErrors(cusparseCreateCsrilu02Info(&infoILU));
  checkCudaErrors(cusparseCreateMatDescr(&matLU));
  checkCudaErrors(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO));

  /* Allocate workspace for cuSPARSE */
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA, vecp,
      &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
      &bufferSizeMV));
  checkCudaErrors(cudaMalloc(&d_bufferMV, bufferSizeMV));

  checkCudaErrors(cusparseScsrilu02_bufferSize(cusparseHandle, N, nz, matLU,
                                               d_val, d_row, d_col, infoILU,
                                               &bufferSizeLU));
  checkCudaErrors(cudaMalloc(&d_bufferLU, bufferSizeLU));

  checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrL));
  checkCudaErrors(cusparseSpSV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower,
      vecR, vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL,
      &bufferSizeL));
  checkCudaErrors(cudaMalloc(&d_bufferL, bufferSizeL));

  checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrU));
  checkCudaErrors(cusparseSpSV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper,
      vecR, vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
      &bufferSizeU));
  checkCudaErrors(cudaMalloc(&d_bufferU, bufferSizeU));

  timer.End();
  ugu::LOGI("prepare %f ms\n", timer.elapsed_msec());
  timer.Start();

  /* Conjugate gradient without preconditioning.
     ------------------------------------------

     Follows the description by Golub & Van Loan,
     "Matrix Computations 3rd ed.", Section 10.2.6  */

  printf("Convergence of CG without preconditioning: \n");
  int k = 0;
  r0 = 0;
  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

  while (r1 > tol * tol && k <= max_iter) {
    k++;

    if (k == 1) {
      checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_p, 1));
    } else {
      beta = r1 / r0;
      checkCudaErrors(cublasSscal(cublasHandle, N, &beta, d_p, 1));
      checkCudaErrors(cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1));
    }

    checkCudaErrors(cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                 matA, vecp, &floatzero, vecomega, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot));
    alpha = r1 / dot;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
    nalpha = -alpha;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
    r0 = r1;
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
  }

  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

  checkCudaErrors(
      cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

  /* check result */
  err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = h_csrRowPtrA[i]; j < h_csrRowPtrA[i + 1]; j++) {
      rsum += h_csrValA[j] * h_x[h_csrColIndA[j]];
    }

    diff = fabs(rsum - h_b[i]);

    if (diff > err) {
      err = diff;
    }
  }

  printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
  nErrors += (k > max_iter) ? 1 : 0;
  qaerr1 = err;

  if (0) {
    // output result in matlab-style array
    int n = (int)sqrt((double)N);
    printf("a = [  ");

    for (int iy = 0; iy < n; iy++) {
      for (int ix = 0; ix < n; ix++) {
        printf(" %f ", h_x[iy * n + ix]);
      }

      if (iy == n - 1) {
        printf(" ]");
      }

      printf("\n");
    }
  }

  timer.End();
  ugu::LOGI("no precond %f ms\n", timer.elapsed_msec());
  timer.Start();

  /* Preconditioned Conjugate Gradient using ILU.
     --------------------------------------------
     Follows the description by Golub & Van Loan,
     "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

  printf("\nConvergence of CG using ILU(0) preconditioning: \n");

  /* Perform analysis for ILU(0) */
  checkCudaErrors(cusparseScsrilu02_analysis(
      cusparseHandle, N, nz, descr, d_valsILU0, d_row, d_col, infoILU,
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

  /* generate the ILU(0) factors */
  checkCudaErrors(
      cusparseScsrilu02(cusparseHandle, N, nz, matLU, d_valsILU0, d_row, d_col,
                        infoILU, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

  /* perform triangular solve analysis */
  checkCudaErrors(
      cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &floatone, matM_lower, vecR, vecX, CUDA_R_32F,
                            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL));

  checkCudaErrors(
      cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &floatone, matM_upper, vecR, vecX, CUDA_R_32F,
                            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU));

  /* reset the initial guess of the solution to zero */
  for (int i = 0; i < N; i++) {
    h_x[i] = 0.0;
  }
  checkCudaErrors(
      cudaMemcpy(d_r, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

  k = 0;
  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

  while (r1 > tol * tol && k <= max_iter) {
    // preconditioner application: d_zm1 = U^-1 L^-1 d_r
    checkCudaErrors(cusparseSpSV_solve(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower,
        vecR, vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

    checkCudaErrors(cusparseSpSV_solve(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper,
        vecY, vecZM1, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));
    k++;

    if (k == 1) {
      checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1));
    } else {
      checkCudaErrors(
          cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
      checkCudaErrors(
          cublasSdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator));
      beta = numerator / denominator;
      checkCudaErrors(cublasSscal(cublasHandle, N, &beta, d_p, 1));
      checkCudaErrors(
          cublasSaxpy(cublasHandle, N, &floatone, d_zm1, 1, d_p, 1));
    }

    checkCudaErrors(cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                 matA, vecp, &floatzero, vecomega, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
    checkCudaErrors(
        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator));
    alpha = numerator / denominator;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
    checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_rm2, 1));
    checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1));
    nalpha = -alpha;
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
  }

  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

  checkCudaErrors(
      cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

  /* check result */
  err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = h_csrRowPtrA[i]; j < h_csrRowPtrA[i + 1]; j++) {
      rsum += h_csrValA[j] * h_x[h_csrColIndA[j]];
    }

    diff = fabs(rsum - h_b[i]);

    if (diff > err) {
      err = diff;
    }
  }

  printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
  nErrors += (k > max_iter) ? 1 : 0;
  qaerr2 = err;

  timer.End();
  ugu::LOGI("with precond %f ms\n", timer.elapsed_msec());
  timer.Start();

  /* Destroy descriptors */
  checkCudaErrors(cusparseDestroyCsrilu02Info(infoILU));
  checkCudaErrors(cusparseDestroyMatDescr(matLU));
  checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrL));
  checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrU));
  checkCudaErrors(cusparseDestroySpMat(matM_lower));
  checkCudaErrors(cusparseDestroySpMat(matM_upper));
  checkCudaErrors(cusparseDestroySpMat(matA));
  checkCudaErrors(cusparseDestroyDnVec(vecp));
  checkCudaErrors(cusparseDestroyDnVec(vecomega));
  checkCudaErrors(cusparseDestroyDnVec(vecR));
  checkCudaErrors(cusparseDestroyDnVec(vecX));
  checkCudaErrors(cusparseDestroyDnVec(vecY));
  checkCudaErrors(cusparseDestroyDnVec(vecZM1));

  /* Destroy contexts */
  checkCudaErrors(cusparseDestroy(cusparseHandle));
  checkCudaErrors(cublasDestroy(cublasHandle));

  /* Free device memory */
  // free(I);
  // free(J);
  // free(val);
  // free(x);
  // free(rhs);
  checkCudaErrors(cudaFree(d_bufferMV));
  checkCudaErrors(cudaFree(d_bufferLU));
  checkCudaErrors(cudaFree(d_bufferL));
  checkCudaErrors(cudaFree(d_bufferU));
  checkCudaErrors(cudaFree(d_col));
  checkCudaErrors(cudaFree(d_row));
  checkCudaErrors(cudaFree(d_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_p));
  checkCudaErrors(cudaFree(d_omega));
  checkCudaErrors(cudaFree(d_valsILU0));
  checkCudaErrors(cudaFree(d_zm1));
  checkCudaErrors(cudaFree(d_zm2));
  checkCudaErrors(cudaFree(d_rm2));

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();

  printf("\n");
  printf("Test Summary:\n");
  printf("   Counted total of %d errors\n", nErrors);
  printf("   qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
  // exit((nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5
  //           ? EXIT_SUCCESS
  //           : EXIT_FAILURE));

  timer.End();
  ugu::LOGI("free %f ms\n", timer.elapsed_msec());
  timer.Start();

  return true;
}

}  // namespace

namespace ugu {

bool SolveSparse(const Eigen::SparseMatrix<double> &mat,
                 const Eigen::MatrixXd &b, Eigen::MatrixXd &x, int devID) {
  Timer<> timer;
  timer.Start();
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double> val;

  EigenSparseToCsr(mat, row, col, val);

  x.resizeLike(b);
  timer.End();

  LOGI("SolveSparse preparation %fms\n", timer.elapsed_msec());

#if 0
  bool ret =
      SolveSparse(mat.rows(), mat.cols(), mat.nonZeros(), val.data(),
                  row.data(), col.data(), b.data(), x.data(), b.cols(), devID);
#else

  if (b.cols() != 1) {
    LOGE("not supported\n");
    return false;
  }

  std::vector<float> val_f;
  for (const auto &v : val) {
    val_f.push_back(v);
  }

  Eigen::MatrixXf b_f = b.cast<float>();
  Eigen::MatrixXf x_f = x.cast<float>();

  bool ret = SolveSparseCg(mat.rows(), mat.cols(), mat.nonZeros(), val_f.data(),
                           row.data(), col.data(), b_f.data(), x_f.data(),
                           b.cols(), devID);
  x = x_f.cast<double>();
#endif

  return ret;
}

bool SolveSparse(int rowsA, int colsA,
                 const std::vector<Eigen::Triplet<double>> &triplets,
                 Eigen::VectorXd b, Eigen::VectorXd &x, int devID) {
  std::vector<Eigen::Triplet<double>> triplets_;
  std::map<std::pair<int, int>, size_t> added_map;
  // Remove duplicate
  for (size_t i = 0; i < triplets.size(); i++) {
    const auto &triplet = triplets[i];
    auto rc = std::make_pair(triplet.row(), triplet.col());
    auto pos = added_map.find(rc);
    if (pos == added_map.end()) {
      triplets_.push_back(triplet);
      added_map[rc] = i;
    } else {
      // Overwrite
      triplets_[added_map.at(rc)] = triplet;
    }
  }

  // Sort by indices
  std::sort(
      triplets_.begin(), triplets_.end(),
      [&](const Eigen::Triplet<double> &a, const Eigen::Triplet<double> &b) {
        if (a.row() != b.row()) {
          return a.row() < b.row();
        } else {
          return a.col() < b.col();
        }
      });

  std::vector<double> csr_vals;
  std::vector<int> csr_row_ptrs;
  std::vector<int> csr_col_indices;

  for (const auto &t : triplets_) {
    // std::cout << t.row() << " " << t.col() << " " << t.value() << std::endl;
    csr_vals.push_back(t.value());
    csr_col_indices.push_back(t.col());
    csr_row_ptrs.push_back(t.row());
  }

  int nnzA = static_cast<int>(csr_vals.size());

  // int count = 0;
  int current_row = 0;
  int current_idx = 0;
  // int current_sum = 0;
  std::vector<int> csr_row_ptrs_;
  // for (int i = 0; i < static_cast<int>(csr_row_ptrs.size()); i++) {
  //   assert(current <= csr_row_ptrs[i]);
  //   if (current < csr_row_ptrs[i]) {
  //     csr_row_ptrs_.push_back(i);
  //   }
  //   //int tmp = current + 1;
  //   //while (tmp <= csr_row_ptrs[i]) {
  //   //  csr_row_ptrs_.push_back(0);
  //   //  tmp++;
  //   //}
  // }
  csr_row_ptrs_.resize(rowsA + 1);
  csr_row_ptrs_[0] = 0;
  while (current_idx < static_cast<int>(csr_row_ptrs.size())) {
    int row = csr_row_ptrs[current_idx];
    if (current_row < row) {
      while (current_row < row) {
        current_row++;
        csr_row_ptrs_[current_row] = current_idx;
      }
    }
    current_idx++;
  }
  csr_row_ptrs_.back() = nnzA;

  csr_row_ptrs = std::move(csr_row_ptrs_);

  x.resizeLike(b);

  for (int i = 0; i < nnzA; i++) {
    //  std::cout << csr_vals[i] << " " << csr_col_indices[i] << std::endl;
  }
  for (int i = 0; i < rowsA + 1; i++) {
    // std::cout << csr_row_ptrs[i] << std::endl;
  }
  // std::cout << b << std::endl;

  return SolveSparse(rowsA, colsA, nnzA, csr_vals.data(), csr_row_ptrs.data(),
                     csr_col_indices.data(), b.data(), x.data(), b.cols(),
                     devID);
}

bool SolveSparse(int rowsA, int colsA, int nnzA, const double *h_csrValA,
                 const int *h_csrRowPtrA, const int *h_csrColIndA,
                 const double *h_b, double *h_x, int out_col, int devID) {
  Timer<> timer;
  timer.Start();
  struct testOpts opts;
  cusolverSpHandle_t handle = NULL;
  cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrA = NULL;

  // int rowsA = 0; /* number of rows of A */
  // int colsA = 0; /* number of columns of A */
  // int nnzA = 0;  /* number of nonzeros of A */
  int baseA = 0; /* base index in CSR format */

  /* CSR(A) from I/O */
  // int *h_csrRowPtrA = NULL;
  // int *h_csrColIndA = NULL;
  // double *h_csrValA = NULL;

  double *h_z = NULL; /* z = B \ (Q*b) */
  // double *h_x = NULL;  /* x = A \ b */
  // double *h_b = NULL;  /* b = ones(n,1) */
  double *h_Qb = NULL; /* Q*b */
  double *h_r = NULL;  /* r = b - A*x */

  int *h_Q = NULL; /* <int> n */
                   /* reorder to reduce zero fill-in */
                   /* Q = symrcm(A) or Q = symamd(A) */
  /* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
  int *h_csrRowPtrB = NULL; /* <int> n+1 */
  int *h_csrColIndB = NULL; /* <int> nnzA */
  double *h_csrValB = NULL; /* <double> nnzA */
  int *h_mapBfromA = NULL;  /* <int> nnzA */

  size_t size_perm = 0;
  void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

  /* device copy of A: used in residual evaluation */
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;

  /* device copy of B: used in B*z = Q*b */
  int *d_csrRowPtrB = NULL;
  int *d_csrColIndB = NULL;
  double *d_csrValB = NULL;

  int *d_Q = NULL;    /* device copy of h_Q */
  double *d_z = NULL; /* z = B \ Q*b */
  // double *d_x = NULL;  /* x = A \ b */
  double *d_b = NULL;  /* a copy of h_b */
  double *d_Qb = NULL; /* a copy of h_Qb */
  double *d_r = NULL;  /* r = b - A*x */

  double tol = 1.e-12;
  const int reorder = 0; /* no reordering */
  int singularity = 0;   /* -1 if A is invertible under tol. */

  /* the constants are used in residual evaluation, r = b - A*x */
  const double minus_one = -1.0;
  const double one = 1.0;

  double b_inf = 0.0;
  double x_inf = 0.0;
  double r_inf = 0.0;
  double A_inf = 0.0;
  int errors = 0;
  int issym = 0;

  double start, stop;
  double time_solve_cpu = 999999;
  double time_solve_gpu;
#if 0
  parseCommandLineArguments(argc, argv, opts);

  if (NULL == opts.testFunc) {
    opts.testFunc = "chol"; /* By default running Cholesky as NO solver selected
                               with -R option. */
  }

  findCudaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
    if (opts.sparse_mat_filename != NULL)
      printf("Using default input file [%s]\n", opts.sparse_mat_filename);
    else
      printf("Could not find lap2D_5pt_n100.mtx\n");
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  printf("step 1: read matrix market format\n");

  if (opts.sparse_mat_filename == NULL) {
    fprintf(stderr, "Error: input matrix is not provided\n");
    return EXIT_FAILURE;
  }

  if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                 &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                 &h_csrColIndA, true)) {
    exit(EXIT_FAILURE);
  }
  baseA = h_csrRowPtrA[0];  // baseA = {0,1}
  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }
#endif
  checkCudaErrors(cusolverSpCreate(&handle));
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  checkCudaErrors(cudaStreamCreate(&stream));
  /* bind stream to cusparse and cusolver*/
  checkCudaErrors(cusolverSpSetStream(handle, stream));
  checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

  /* configure matrix descriptor*/
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  if (baseA) {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
  } else {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  }

  h_z = (double *)malloc(sizeof(double) * colsA * out_col);
  // h_x = (double *)malloc(sizeof(double) * colsA);
  // h_b = (double *)malloc(sizeof(double) * rowsA);
  h_Qb = (double *)malloc(sizeof(double) * rowsA * out_col);
  h_r = (double *)malloc(sizeof(double) * rowsA);

  h_Q = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB = (double *)malloc(sizeof(double) * nnzA);
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

  assert(NULL != h_z);
  assert(NULL != h_x);
  assert(NULL != h_b);
  assert(NULL != h_Qb);
  assert(NULL != h_r);
  assert(NULL != h_Q);
  assert(NULL != h_csrRowPtrB);
  assert(NULL != h_csrColIndB);
  assert(NULL != h_csrValB);
  assert(NULL != h_mapBfromA);

  checkCudaErrors(
      cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkCudaErrors(
      cudaMalloc((void **)&d_csrRowPtrB, sizeof(int) * (rowsA + 1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndB, sizeof(int) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValB, sizeof(double) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_Q, sizeof(int) * colsA));
  checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(double) * colsA * out_col));
  // checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA ));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * rowsA * out_col));
  checkCudaErrors(cudaMalloc((void **)&d_Qb, sizeof(double) * rowsA * out_col));
  checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

  timer.End();
  LOGI("malloc %f ms\n", timer.elapsed_msec());
  timer.Start();

  /* verify if A has symmetric pattern or not */
  checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
                                          h_csrRowPtrA, h_csrRowPtrA + 1,
                                          h_csrColIndA, &issym));
  opts.testFunc = "chol";
  opts.reorder = "metis";

  if (0 == strcmp(opts.testFunc, "chol")) {
    if (!issym) {
      printf("Error: A has no symmetric pattern, please use LU or QR \n");
      exit(EXIT_FAILURE);
    }
  }

  timer.End();
  LOGI("symmetric test %f ms\n", timer.elapsed_msec());
  timer.Start();

  // printf("step 2: reorder the matrix A to minimize zero fill-in\n");
  // printf(
  //     "        if the user choose a reordering by -P=symrcm, -P=symamd or "
  //     "-P=metis\n");

  if (NULL != opts.reorder) {
    if (0 == strcmp(opts.reorder, "symrcm")) {
      //  printf("step 2.1: Q = symrcm(A) \n");
      checkCudaErrors(cusolverSpXcsrsymrcmHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(opts.reorder, "symamd")) {
      //  printf("step 2.1: Q = symamd(A) \n");
      checkCudaErrors(cusolverSpXcsrsymamdHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(opts.reorder, "metis")) {
      //   printf("step 2.1: Q = metis(A) \n");
      checkCudaErrors(cusolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
                                                h_csrRowPtrA, h_csrColIndA,
                                                NULL, /* default setting. */
                                                h_Q));
    } else {
      fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
      return 1;
    }
  } else {
    printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
    for (int j = 0; j < rowsA; j++) {
      h_Q[j] = j;
    }
  }

  // printf("step 2.2: B = A(Q,Q) \n");

  memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
  memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

  checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
      handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
      &size_perm));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(NULL != buffer_cpu);

  /* h_mapBfromA = Identity */
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA[j] = j;
  }
  checkCudaErrors(cusolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
                                         h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
                                         h_mapBfromA, buffer_cpu));

  /* B = A( mapBfromA ) */
  for (int j = 0; j < nnzA; j++) {
    h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
  }
#if 1
  // printf("step 3: b(j) = 1 + j/n \n");
  // for (int row = 0; row < rowsA; row++) {
  //   h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
  // }

  /* h_Qb = b(Q) */
  for (int col = 0; col < out_col; col++) {
    for (int row = 0; row < rowsA; row++) {
      int index0 = rowsA * col + row;
      int index1 = rowsA * col + h_Q[row];
      h_Qb[index0] = h_b[index1];
    }
  }
#endif

  timer.End();
  LOGI("host reorder %f ms\n", timer.elapsed_msec());
  timer.Start();

  // printf("step 4: prepare data on device\n");
  checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA,
                                  sizeof(int) * (rowsA + 1),
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA,
                                  sizeof(int) * nnzA, cudaMemcpyHostToDevice,
                                  stream));
  checkCudaErrors(cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB,
                                  sizeof(int) * (rowsA + 1),
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_csrColIndB, h_csrColIndB,
                                  sizeof(int) * nnzA, cudaMemcpyHostToDevice,
                                  stream));
  checkCudaErrors(cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA * out_col,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA * out_col,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA,
                                  cudaMemcpyHostToDevice, stream));

  timer.End();
  LOGI("cuda memcpy %f ms\n", timer.elapsed_msec());
  timer.Start();

#if 0
  printf("step 5: solve A*x = b on CPU \n");
  start = second();

  /* solve B*z = Q*b */
  if (0 == strcmp(opts.testFunc, "chol")) {
    checkCudaErrors(cusolverSpDcsrlsvcholHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));
  } else if (0 == strcmp(opts.testFunc, "lu")) {
    checkCudaErrors(cusolverSpDcsrlsvluHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));

  } else if (0 == strcmp(opts.testFunc, "qr")) {
    checkCudaErrors(cusolverSpDcsrlsvqrHost(
        handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        h_Qb, tol, reorder, h_z, &singularity));
  } else {
    fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
    return 1;
  }

  /* Q*x = z */
  for (int row = 0; row < rowsA; row++) {
    h_x[h_Q[row]] = h_z[row];
  }

  if (0 <= singularity) {
    printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
           singularity, tol);
  }

  stop = second();
  time_solve_cpu = stop - start;

  printf("step 6: evaluate residual r = b - A*x (result on CPU)\n");
  checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
                                  cudaMemcpyDeviceToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_x, h_x, sizeof(double) * colsA,
                                  cudaMemcpyHostToDevice, stream));
#endif

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  if (baseA) {
    checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F));
  } else {
    checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
                                      d_csrColIndA, d_csrValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  }
  timer.End();
  LOGI("cuspase prepare %f ms\n", timer.elapsed_msec());
  timer.Start();

#if 0
  cusparseDnVecDescr_t vecx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F));
  cusparseDnVecDescr_t vecAx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F));

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
      &one, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));
 
  checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
                                  cudaMemcpyDeviceToHost, stream));
  /* wait until h_r is ready */
  checkCudaErrors(cudaDeviceSynchronize());
#endif

#if 0
  b_inf = vec_norminf(rowsA, h_b);
  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);
  A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
                          h_csrColIndA);

  printf("(CPU) |b - A*x| = %E \n", r_inf);
  printf("(CPU) |A| = %E \n", A_inf);
  printf("(CPU) |x| = %E \n", x_inf);
  printf("(CPU) |b| = %E \n", b_inf);
  printf("(CPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
         r_inf / (A_inf * x_inf + b_inf));

#endif

  // printf("step 7: solve A*x = b on GPU\n");
  start = second();

  /* solve B*z = Q*b */
  for (int col = 0; col < out_col; col++) {
    int offset = rowsA * col;
    if (0 == strcmp(opts.testFunc, "chol")) {
      checkCudaErrors(cusolverSpDcsrlsvchol(
          handle, rowsA, nnzA, descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
          d_Qb + offset, tol, reorder, d_z + offset, &singularity));

    } else if (0 == strcmp(opts.testFunc, "lu")) {
      printf("WARNING: no LU available on GPU \n");
    } else if (0 == strcmp(opts.testFunc, "qr")) {
      checkCudaErrors(cusolverSpDcsrlsvqr(
          handle, rowsA, nnzA, descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
          d_Qb + offset, tol, reorder, d_z + offset, &singularity));
    } else {
      fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
      return 1;
    }
  }
  timer.End();
  LOGI("solve without waiting %f ms\n", timer.elapsed_msec());
  timer.Start();
  checkCudaErrors(cudaDeviceSynchronize());
  if (0 <= singularity) {
    printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
           singularity, tol);
  }
  timer.End();
  LOGI("solve %f ms\n", timer.elapsed_msec());
  timer.Start();

#if 0
  /* Q*x = z */
  cusparseSpVecDescr_t vecz = NULL;
  checkCudaErrors(cusparseCreateSpVec(&vecz, colsA, rowsA, d_Q, d_z,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseScatter(cusparseHandle, vecz, vecx));
  checkCudaErrors(cusparseDestroySpVec(vecz));
#endif
  cudaMemcpyAsync(h_z, d_z, sizeof(double) * colsA * out_col,
                  cudaMemcpyDeviceToHost, stream);

  checkCudaErrors(cudaDeviceSynchronize());

  stop = second();
  time_solve_gpu = stop - start;
#if 0
  printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
  checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
                                  cudaMemcpyDeviceToDevice, stream));

  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA,
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
                                  cudaMemcpyDeviceToHost, stream));
  /* wait until h_x and h_r are ready */
  checkCudaErrors(cudaDeviceSynchronize());

  b_inf = vec_norminf(rowsA, h_b);
  x_inf = vec_norminf(colsA, h_x);
  r_inf = vec_norminf(rowsA, h_r);

  if (0 != strcmp(opts.testFunc, "lu")) {
    // only cholesky and qr have GPU version
    printf("(GPU) |b - A*x| = %E \n", r_inf);
    printf("(GPU) |A| = %E \n", A_inf);
    printf("(GPU) |x| = %E \n", x_inf);
    printf("(GPU) |b| = %E \n", b_inf);
    printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
           r_inf / (A_inf * x_inf + b_inf));
  }

  fprintf(stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n",
          opts.testFunc, time_solve_cpu, time_solve_gpu);

  if (0 != strcmp(opts.testFunc, "lu")) {
    printf("show last 10 elements of solution vector (GPU) \n");
    printf("consistent result for different reordering and solver \n");
    for (int j = rowsA - 10; j < rowsA; j++) {
      printf("x[%d] = %E\n", j, h_x[j]);
    }
  }
#endif

  /* Q*x = z */

  for (int col = 0; col < out_col; col++) {
    for (int row = 0; row < rowsA; row++) {
      int index0 = rowsA * col + row;
      int index1 = rowsA * col + h_Q[row];
      h_x[index1] = h_z[index0];
    }
  }

  timer.End();
  LOGI("copy to host %f ms\n", timer.elapsed_msec());
  timer.Start();

  if (handle) {
    checkCudaErrors(cusolverSpDestroy(handle));
  }
  if (cusparseHandle) {
    checkCudaErrors(cusparseDestroy(cusparseHandle));
  }
  if (stream) {
    checkCudaErrors(cudaStreamDestroy(stream));
  }
  if (descrA) {
    checkCudaErrors(cusparseDestroyMatDescr(descrA));
  }
  if (matA) {
    checkCudaErrors(cusparseDestroySpMat(matA));
  }
  // if (vecx) {
  //   checkCudaErrors(cusparseDestroyDnVec(vecx));
  // }
  // if (vecAx) {
  //   checkCudaErrors(cusparseDestroyDnVec(vecAx));
  // }

  // if (h_csrValA) {
  //   free(h_csrValA);
  // }
  // if (h_csrRowPtrA) {
  //   free(h_csrRowPtrA);
  // }
  // if (h_csrColIndA) {
  //   free(h_csrColIndA);
  // }
  if (h_z) {
    free(h_z);
  }
  // if (h_x) {
  //   free(h_x);
  // }
  // if (h_b) {
  //   free(h_b);
  // }
  if (h_Qb) {
    free(h_Qb);
  }
  if (h_r) {
    free(h_r);
  }

  if (h_Q) {
    free(h_Q);
  }

  if (h_csrRowPtrB) {
    free(h_csrRowPtrB);
  }
  if (h_csrColIndB) {
    free(h_csrColIndB);
  }
  if (h_csrValB) {
    free(h_csrValB);
  }
  if (h_mapBfromA) {
    free(h_mapBfromA);
  }

  if (buffer_cpu) {
    free(buffer_cpu);
  }

  // if (buffer) {
  //   checkCudaErrors(cudaFree(buffer));
  // }
  if (d_csrValA) {
    checkCudaErrors(cudaFree(d_csrValA));
  }
  if (d_csrRowPtrA) {
    checkCudaErrors(cudaFree(d_csrRowPtrA));
  }
  if (d_csrColIndA) {
    checkCudaErrors(cudaFree(d_csrColIndA));
  }
  if (d_csrValB) {
    checkCudaErrors(cudaFree(d_csrValB));
  }
  if (d_csrRowPtrB) {
    checkCudaErrors(cudaFree(d_csrRowPtrB));
  }
  if (d_csrColIndB) {
    checkCudaErrors(cudaFree(d_csrColIndB));
  }
  if (d_Q) {
    checkCudaErrors(cudaFree(d_Q));
  }
  if (d_z) {
    checkCudaErrors(cudaFree(d_z));
  }
  // if (d_x) {
  //   checkCudaErrors(cudaFree(d_x));
  // }
  if (d_b) {
    checkCudaErrors(cudaFree(d_b));
  }
  if (d_Qb) {
    checkCudaErrors(cudaFree(d_Qb));
  }
  if (d_r) {
    checkCudaErrors(cudaFree(d_r));
  }

  timer.End();
  LOGI("free %f ms\n", timer.elapsed_msec());
  timer.Start();

  return true;
}

}  // namespace ugu
