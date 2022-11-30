// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Sambit Das
//

#include <constants.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include "deviceHelpers.h"
#include <cuComplex.h>
#include "linearAlgebraOperationsDevice.h"

namespace dftfe
{
  namespace
  {
    template <typename NumberType>
    __global__ void
    stridedCopyToBlockKernel(const unsigned int BVec,
                             const NumberType * xVec,
                             const unsigned int M,
                             const unsigned int N,
                             NumberType *       yVec,
                             const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries  = M * BVec;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex      = index / BVec;
          unsigned int intraBlockIndex = index - blockIndex * BVec;
          yVec[index] = xVec[blockIndex * N + startingXVecId + intraBlockIndex];
        }
    }

    __global__ void
    copyGlobalToCellDeviceKernel(const unsigned int contiguousBlockSize,
                                 const unsigned int numContiguousBlocks,
                                 const double *     copyFromVec,
                                 double *           copyToVec,
                                 const dealii::types::global_dof_index
                                   *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          copyToVec[index] =
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex];
        }
    }

    __global__ void
    copyGlobalToCellDeviceKernel(const unsigned int contiguousBlockSize,
                                 const unsigned int numContiguousBlocks,
                                 const double *     copyFromVec,
                                 float *            copyToVec,
                                 const dealii::types::global_dof_index
                                   *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          copyToVec[index] =
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex];
        }
    }

    __global__ void
    copyGlobalToCellDeviceKernel(const unsigned int     contiguousBlockSize,
                                 const unsigned int     numContiguousBlocks,
                                 const cuDoubleComplex *copyFromVec,
                                 cuDoubleComplex *      copyToVec,
                                 const dealii::types::global_dof_index
                                   *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          copyToVec[index] =
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex];
        }
    }

    __global__ void
    copyGlobalToCellDeviceKernel(const unsigned int     contiguousBlockSize,
                                 const unsigned int     numContiguousBlocks,
                                 const cuDoubleComplex *copyFromVec,
                                 cuFloatComplex *       copyToVec,
                                 const dealii::types::global_dof_index
                                   *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          copyToVec[index] = make_cuFloatComplex(
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]
              .x,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]
              .y);
        }
    }

    __global__ void
    copyDeviceKernel(const unsigned int size,
                     const double *     copyFromVec,
                     double *           copyToVec)
    {
      for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < size;
           index += blockDim.x * gridDim.x)
        copyToVec[index] = copyFromVec[index];
    }

    __global__ void
    copyDeviceKernel(const unsigned int size,
                     const double *     copyFromVec,
                     cuDoubleComplex *  copyToVec)
    {
      for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < size;
           index += blockDim.x * gridDim.x)
        {
          copyToVec[index] = make_cuDoubleComplex(copyFromVec[index], 0.0);
        }
    }

    __global__ void
    copyDeviceKernel(const unsigned int size,
                     const double *     copyFromVec,
                     float *            copyToVec)
    {
      for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < size;
           index += blockDim.x * gridDim.x)
        copyToVec[index] = copyFromVec[index];
    }

    __global__ void
    copyDeviceKernel(const unsigned int size,
                     const double *     copyFromVec,
                     cuFloatComplex *   copyToVec)
    {
      for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
           index < size;
           index += blockDim.x * gridDim.x)
        {
          copyToVec[index] = make_cuFloatComplex(copyFromVec[index], 0.0);
        }
    }

    void
    copyDoubleToNumber(const double *     copyFromVec,
                       const unsigned int size,
                       double *           copyToVec)
    {
      copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
    }

    void
    copyDoubleToNumber(const double *     copyFromVec,
                       const unsigned int size,
                       cuDoubleComplex *  copyToVec)
    {
      copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
    }


    void
    copyDoubleToNumber(const double *     copyFromVec,
                       const unsigned int size,
                       float *            copyToVec)
    {
      copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
    }

    void
    copyDoubleToNumber(const double *     copyFromVec,
                       const unsigned int size,
                       cuFloatComplex *   copyToVec)
    {
      copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             double *           XQuads,
                                             double *           XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi      = XQuads[index];
          const double psiPrime = XPrimeQuads[index];
          XPrimeQuads[index]    = psi * psiPrime;
          XQuads[index]         = psi * psi;
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             cuDoubleComplex *  XQuads,
                                             cuDoubleComplex *  XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const cuDoubleComplex psi      = XQuads[index];
          const cuDoubleComplex psiPrime = XPrimeQuads[index];
          XPrimeQuads[index] =
            make_cuDoubleComplex(psi.x * psiPrime.x + psi.y * psiPrime.y, 0.0);
          XQuads[index] =
            make_cuDoubleComplex(psi.x * psi.x + psi.y * psi.y, 0.0);
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             float *            XQuads,
                                             float *            XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const float psi      = XQuads[index];
          const float psiPrime = XPrimeQuads[index];
          XPrimeQuads[index]   = psi * psiPrime;
          XQuads[index]        = psi * psi;
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             cuFloatComplex *   XQuads,
                                             cuFloatComplex *   XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const cuFloatComplex psi      = XQuads[index];
          const cuFloatComplex psiPrime = XPrimeQuads[index];
          XPrimeQuads[index] =
            make_cuFloatComplex(psi.x * psiPrime.x + psi.y * psiPrime.y, 0.0);
          XQuads[index] =
            make_cuFloatComplex(psi.x * psi.x + psi.y * psi.y, 0.0);
        }
    }
  } // namespace

  template <typename NumberType, typename NumberTypeLowPrec>
  void
  computeRhoFirstOrderResponseDevice(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
    cudaDeviceSynchronize();
    MPI_Barrier(mpiCommParent);
    double             device_time = MPI_Wtime();
    const unsigned int numKPoints  = kPointWeights.size();

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int BVec =
      std::min(dftParams.chebyWfcBlockSize, totalNumWaveFunctions);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;

    const NumberTypeLowPrec zero =
      deviceUtils::makeNumberFromReal<NumberTypeLowPrec>(0.0);
    const NumberTypeLowPrec one =
      deviceUtils::makeNumberFromReal<NumberTypeLowPrec>(1.0);
    const NumberTypeLowPrec scalarCoeffAlphaRho =
      deviceUtils::makeNumberFromReal<NumberTypeLowPrec>(1.0);
    const NumberTypeLowPrec scalarCoeffBetaRho =
      deviceUtils::makeNumberFromReal<NumberTypeLowPrec>(1.0);

    const unsigned int cellsBlockSize = 50;
    const unsigned int numCellBlocks  = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      rhoResponseContributionHamDevice(totalLocallyOwnedCells * numQuadPoints,
                                       zero);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      rhoResponseContributionFermiEnergyDevice(totalLocallyOwnedCells *
                                                 numQuadPoints,
                                               zero);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Host>
      rhoResponseContributionHamHost(totalLocallyOwnedCells * numQuadPoints,
                                     zero);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Host>
      rhoResponseContributionFermiEnergyHost(totalLocallyOwnedCells *
                                               numQuadPoints,
                                             zero);

    std::vector<double> rhoResponseValuesHamFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints, 0.0);
    std::vector<double> rhoResponseValuesFermiEnergyFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints, 0.0);

    std::vector<double> rhoResponseValuesSpinPolarizedHamFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);
    std::vector<double> rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      XQuadsDevice(cellsBlockSize * numQuadPoints * BVec, zero);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      XPrimeQuadsDevice(cellsBlockSize * numQuadPoints * BVec, zero);
    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      onesVecDevice(BVec, one);

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Host>
      densityMatDerFermiEnergyVec(BVec, zero);
    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      densityMatDerFermiEnergyVecDevice(BVec, zero);

    distributedDeviceVec<NumberType> &deviceFlattenedArrayXBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedDeviceVec<NumberType> &deviceFlattenedArrayXPrimeBlock =
      operatorMatrix.getParallelChebyBlockVector2Device();

    const unsigned int numGhosts =
      deviceFlattenedArrayXBlock.ghostFlattenedSize();

    // NumberType *cellWaveFunctionMatrix = reinterpret_cast<NumberType *>(
    //  thrust::raw_pointer_cast(&operatorMatrix.getCellWaveFunctionMatrix()[0]));

    deviceUtils::Vector<NumberTypeLowPrec, dftfe::MemorySpace::Device>
      cellWaveFunctionMatrix(cellsBlockSize * numNodesPerElement * BVec, zero);

    NumberTypeLowPrec *shapeFunctionValuesTransposedDevice;

    DeviceCHECK(cudaMalloc((void **)&shapeFunctionValuesTransposedDevice,
                           numNodesPerElement * numQuadPoints *
                             sizeof(NumberTypeLowPrec)));
    DeviceCHECK(cudaMemset(shapeFunctionValuesTransposedDevice,
                           0,
                           numNodesPerElement * numQuadPoints *
                             sizeof(NumberTypeLowPrec)));

    copyDoubleToNumber(thrust::raw_pointer_cast(
                         &(operatorMatrix.getShapeFunctionValuesTransposed(
                           true)[0])),
                       numNodesPerElement * numQuadPoints,
                       shapeFunctionValuesTransposedDevice);

    for (unsigned int spinIndex = 0; spinIndex < (1 + dftParams.spinPolarized);
         ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            rhoResponseContributionHamDevice.set(zero);
            rhoResponseContributionFermiEnergyDevice.set(zero);

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                         ++iEigenVec)
                      {
                        *(densityMatDerFermiEnergyVec.begin() + iEigenVec) =
                          deviceUtils::makeNumberFromReal<NumberTypeLowPrec>(
                            densityMatDerFermiEnergy
                              [(dftParams.spinPolarized + 1) * kPoint +
                               spinIndex][jvec + iEigenVec]);
                      }

                    deviceUtils::copyHostVecToDeviceVec(
                      densityMatDerFermiEnergyVec.begin(),
                      densityMatDerFermiEnergyVecDevice.begin(),
                      densityMatDerFermiEnergyVecDevice.size());

                    stridedCopyToBlockKernel<<<
                      (BVec + 255) / 256 * numLocalDofs,
                      256>>>(BVec,
                             X + numLocalDofs * totalNumWaveFunctions *
                                   ((dftParams.spinPolarized + 1) * kPoint +
                                    spinIndex),
                             numLocalDofs,
                             totalNumWaveFunctions,
                             deviceFlattenedArrayXBlock.begin(),
                             jvec);


                    deviceFlattenedArrayXBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(deviceFlattenedArrayXBlock, BVec);

                    stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                 numLocalDofs,
                                               256>>>(
                      BVec,
                      XPrime +
                        numLocalDofs * totalNumWaveFunctions *
                          ((dftParams.spinPolarized + 1) * kPoint + spinIndex),
                      numLocalDofs,
                      totalNumWaveFunctions,
                      deviceFlattenedArrayXPrimeBlock.begin(),
                      jvec);


                    deviceFlattenedArrayXPrimeBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(deviceFlattenedArrayXPrimeBlock, BVec);


                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;


                            copyGlobalToCellDeviceKernel<<<
                              (BVec + 255) / 256 * currentCellsBlockSize *
                                numNodesPerElement,
                              256>>>(
                              BVec,
                              currentCellsBlockSize * numNodesPerElement,
                              deviceFlattenedArrayXBlock.begin(),
                              cellWaveFunctionMatrix.begin(),
                              thrust::raw_pointer_cast(
                                &(operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));

                            NumberTypeLowPrec scalarCoeffAlpha =
                              deviceUtils::makeNumberFromReal<
                                NumberTypeLowPrec>(1.0);
                            NumberTypeLowPrec scalarCoeffBeta =
                              deviceUtils::makeNumberFromReal<
                                NumberTypeLowPrec>(0.0);
                            int strideA = BVec * numNodesPerElement;
                            int strideB = 0;
                            int strideC = BVec * numQuadPoints;


                            cublasXgemmStridedBatched(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              BVec,
                              numQuadPoints,
                              numNodesPerElement,
                              &scalarCoeffAlpha,
                              cellWaveFunctionMatrix.begin(),
                              BVec,
                              strideA,
                              shapeFunctionValuesTransposedDevice,
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);

                            copyGlobalToCellDeviceKernel<<<
                              (BVec + 255) / 256 * currentCellsBlockSize *
                                numNodesPerElement,
                              256>>>(
                              BVec,
                              currentCellsBlockSize * numNodesPerElement,
                              deviceFlattenedArrayXPrimeBlock.begin(),
                              cellWaveFunctionMatrix.begin(),
                              thrust::raw_pointer_cast(
                                &(operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));


                            cublasXgemmStridedBatched(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              BVec,
                              numQuadPoints,
                              numNodesPerElement,
                              &scalarCoeffAlpha,
                              cellWaveFunctionMatrix.begin(),
                              BVec,
                              strideA,
                              shapeFunctionValuesTransposedDevice,
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XPrimeQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);


                            computeRhoResponseFromInterpolatedValues<<<
                              (BVec + 255) / 256 * numQuadPoints *
                                currentCellsBlockSize,
                              256>>>(BVec * numQuadPoints *
                                       currentCellsBlockSize,
                                     XQuadsDevice.begin(),
                                     XPrimeQuadsDevice.begin());

                            cublasXgemm(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              1,
                              currentCellsBlockSize * numQuadPoints,
                              BVec,
                              &scalarCoeffAlphaRho,
                              onesVecDevice.begin(),
                              1,
                              XPrimeQuadsDevice.begin(),
                              BVec,
                              &scalarCoeffBetaRho,
                              rhoResponseContributionHamDevice.begin() +
                                startingCellId * numQuadPoints,
                              1);

                            cublasXgemm(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              1,
                              currentCellsBlockSize * numQuadPoints,
                              BVec,
                              &scalarCoeffAlphaRho,
                              densityMatDerFermiEnergyVecDevice.begin(),
                              1,
                              XQuadsDevice.begin(),
                              BVec,
                              &scalarCoeffBetaRho,
                              rhoResponseContributionFermiEnergyDevice.begin() +
                                startingCellId * numQuadPoints,
                              1);

                          } // non-trivial cell block check
                      }     // cells block loop
                  }         // band parallelizatoin check
              }             // wave function block loop


            // do cuda memcopy to host
            deviceUtils::copyDeviceVecToHostVec(
              rhoResponseContributionHamDevice.begin(),
              rhoResponseContributionHamHost.begin(),
              totalLocallyOwnedCells * numQuadPoints);

            deviceUtils::copyDeviceVecToHostVec(
              rhoResponseContributionFermiEnergyDevice.begin(),
              rhoResponseContributionFermiEnergyHost.begin(),
              totalLocallyOwnedCells * numQuadPoints);

            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoResponseValuesHamFlattenedHost[icell * numQuadPoints +
                                                    iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    deviceUtils::makeRealFromNumber(
                      *(rhoResponseContributionHamHost.begin() +
                        icell * numQuadPoints + iquad));

                  rhoResponseValuesFermiEnergyFlattenedHost[icell *
                                                              numQuadPoints +
                                                            iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    deviceUtils::makeRealFromNumber(
                      *(rhoResponseContributionFermiEnergyHost.begin() +
                        icell * numQuadPoints + iquad));
                }


            if (dftParams.spinPolarized == 1)
              {
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      rhoResponseValuesSpinPolarizedHamFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        deviceUtils::makeRealFromNumber(
                          *(rhoResponseContributionHamHost.begin() +
                            icell * numQuadPoints + iquad));

                      rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        deviceUtils::makeRealFromNumber(
                          *(rhoResponseContributionFermiEnergyHost.begin() +
                            icell * numQuadPoints + iquad));
                    }
              }


          } // kpoint loop
      }     // spin index loop

    // gather density from all inter communicators
    if (dealii::Utilities::MPI::n_mpi_processes(interpoolcomm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseValuesHamFlattenedHost,
                                    interpoolcomm,
                                    rhoResponseValuesHamFlattenedHost);

        dealii::Utilities::MPI::sum(rhoResponseValuesFermiEnergyFlattenedHost,
                                    interpoolcomm,
                                    rhoResponseValuesFermiEnergyFlattenedHost);

        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedHamFlattenedHost,
              interpoolcomm,
              rhoResponseValuesSpinPolarizedHamFlattenedHost);

            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost,
              interpoolcomm,
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost);
          }
      }

    if (dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseValuesHamFlattenedHost,
                                    interBandGroupComm,
                                    rhoResponseValuesHamFlattenedHost);

        dealii::Utilities::MPI::sum(rhoResponseValuesFermiEnergyFlattenedHost,
                                    interBandGroupComm,
                                    rhoResponseValuesFermiEnergyFlattenedHost);

        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedHamFlattenedHost,
              interBandGroupComm,
              rhoResponseValuesSpinPolarizedHamFlattenedHost);

            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost,
              interBandGroupComm,
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost);
          }
      }

    unsigned int                                         iElem = 0;
    typename dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandler.begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::vector<double> &temp1Quads = (rhoResponseValuesHam)[cellid];
          std::vector<double> &temp2Quads =
            (rhoResponseValuesFermiEnergy)[cellid];
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              temp1Quads[q] =
                rhoResponseValuesHamFlattenedHost[iElem * numQuadPoints + q];
              temp2Quads[q] =
                rhoResponseValuesFermiEnergyFlattenedHost[iElem *
                                                            numQuadPoints +
                                                          q];
            }

          if (dftParams.spinPolarized == 1)
            {
              std::vector<double> &temp3Quads =
                (rhoResponseValuesHamSpinPolarized)[cellid];

              std::vector<double> &temp4Quads =
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid];

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  temp3Quads[2 * q + 0] =
                    rhoResponseValuesSpinPolarizedHamFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 0];
                  temp3Quads[2 * q + 1] =
                    rhoResponseValuesSpinPolarizedHamFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 1];
                  temp4Quads[2 * q + 0] =
                    rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 0];
                  temp4Quads[2 * q + 1] =
                    rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 1];
                }
            }

          iElem++;
        }

    DeviceCHECK(cudaFree(shapeFunctionValuesTransposedDevice));
    cudaDeviceSynchronize();
    MPI_Barrier(mpiCommParent);
    device_time = MPI_Wtime() - device_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      std::cout << "Time for compute rhoprime on Device: " << device_time
                << std::endl;
  }

  template void
  computeRhoFirstOrderResponseDevice<dataTypes::numberDevice,
                                     dataTypes::numberDevice>(
    const dataTypes::numberDevice *                X,
    const dataTypes::numberDevice *                XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);

  template void
  computeRhoFirstOrderResponseDevice<dataTypes::numberDevice,
                                     dataTypes::numberFP32Device>(
    const dataTypes::numberDevice *                X,
    const dataTypes::numberDevice *                XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
} // namespace dftfe
