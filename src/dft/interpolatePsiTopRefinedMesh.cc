#include <dft.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <DataTypeOverloads.h>
#include <linearAlgebraOperationsDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h>


namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::interpolatePsiTopRefinedMesh(
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &Psi,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      &                                      basisOperationsPtrSrc,
    const std::vector<double> &              kPointWeights,
    const unsigned int                       totalNumWaveFunctions,
    const unsigned int                       quadratureIndex,
    const dealii::DoFHandler<3> &            dofHandlerDst,
    const dealii::AffineConstraints<double> &constraintMatrixDst,
    const MPI_Comm &                         mpiCommParent,
    const MPI_Comm &                         interpoolcomm,
    const MPI_Comm &                         interBandGroupComm,
    const dftParameters &                    dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    double             interpolate_time = MPI_Wtime();
    const unsigned int numKPoints       = kPointWeights.size();
    const unsigned int numLocalDofs     = basisOperationsPtrSrc->nOwnedDofs();
    const unsigned int totalLocallyOwnedCells = basisOperationsPtrSrc->nCells();
    const unsigned int numNodesPerElement =
      basisOperationsPtrSrc->nDofsPerCell();
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
      std::min(dftParams.chebyWfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1 || dftParams.noncolin || dftParams.hasSOC) ?
        1.0 :
        2.0;
    const unsigned int numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;
    const unsigned int numRhoComponents =
      dftParams.noncolin ? 4 : numSpinComponents;

    const unsigned int numWfnSpinors =
      (dftParams.noncolin || dftParams.hasSOC) ? 2 : 1;

    const dataTypes::number zero                    = 0;
    const dataTypes::number scalarCoeffAlphaRho     = 1.0;
    const dataTypes::number scalarCoeffBetaRho      = 1.0;
    const dataTypes::number scalarCoeffAlphaGradRho = 1.0;
    const dataTypes::number scalarCoeffBetaGradRho  = 1.0;

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtrSrc->reinit(BVec * numWfnSpinors,
                                  cellsBlockSize,
                                  quadratureIndex);
    const unsigned int numQuadPoints = basisOperationsPtrSrc->nQuadsPerCell();

    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      wfcQuadPointData;
    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec *
                              numWfnSpinors,
                            zero);


    const unsigned int numLocalDofsDst = dofHandlerDst.n_locally_owned_dofs();
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> PsiDst(
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size() *
        totalNumWaveFunctions * numLocalDofsDst * numWfnSpinors,
      0.0);
    dealii::MatrixFree<3, double>                  matrixFreeDataTemp;
    typename dealii::MatrixFree<3>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      dealii::MatrixFree<3>::AdditionalData::partition_partition;
    dealii::QGaussLobatto<1> quadrature(FEOrder + 1);
    matrixFreeDataTemp.reinit(dealii::MappingQ1<3, 3>(),
                              dofHandlerDst,
                              constraintMatrixDst,
                              quadrature,
                              additional_data);
    dftUtils::constraintMatrixInfo<memorySpace> constraintInfoDst;
    constraintInfoDst.initialize(matrixFreeDataTemp.get_vector_partitioner(),
                                 constraintMatrixDst);

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlock;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlockDst;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      flattenedArrayBlockDstBVec;

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      flattenedArrayBlockDstBVecRem;
    createMultiVectorFromDealiiPartitioner(
      matrixFreeDataTemp.get_vector_partitioner(),
      BVec * numWfnSpinors,
      flattenedArrayBlockDstBVec);
    flattenedArrayBlockDstBVecRem.reinit(
      flattenedArrayBlockDstBVec.getMPIPatternP2P(),
      (totalNumWaveFunctions % BVec) * numWfnSpinors);


    const dealii::Quadrature<3> &quadrature_formula =
      matrixFreeDataTemp.get_quadrature();
    const unsigned int                   numDofsDst = quadrature_formula.size();
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      dofHandlerDst.get_fe().get_unit_support_points();
    std::vector<unsigned int> renumberingMap(numDofsDst);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (unsigned int i = 0; i < numDofsDst; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (unsigned int j = 0; j < numDofsDst; ++j)
          {
            const dealii::Point<3> &quadCoor = quadraturePointCoor[j];
            double                  dist     = quadCoor.distance(nodalCoor);
            if (dist <= 1e-08)
              {
                renumberingMap[i] = j;
                break;
              }
          }
      }
    std::vector<dealii::types::global_dof_index>
      flattenedArrayCellLocalProcIndexIdMap;
    flattenedArrayCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells *
                                                 numDofsDst);


    auto cellPtr = matrixFreeDataTemp.get_dof_handler().begin_active();
    auto endcPtr = matrixFreeDataTemp.get_dof_handler().end();

    std::vector<global_size_type> cellDofIndicesGlobal(numDofsDst);

    unsigned int iCell = 0;
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          cellPtr->get_dof_indices(cellDofIndicesGlobal);
          for (unsigned int iDof = 0; iDof < numDofsDst; ++iDof)
            flattenedArrayCellLocalProcIndexIdMap[iCell * numDofsDst + iDof] =
              matrixFreeDataTemp.get_vector_partitioner()->global_to_local(
                cellDofIndicesGlobal[iDof]);

          ++iCell;
        }

    std::vector<dftfe::global_size_type>
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvecHost(
        flattenedArrayCellLocalProcIndexIdMap.size());
    std::vector<dftfe::global_size_type>
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRemHost(
        flattenedArrayCellLocalProcIndexIdMap.size());
    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      for (unsigned int iDoF = 0; iDoF < numDofsDst; ++iDoF)
        {
          flattenedArrayCellLocalProcIndexIdMapRenumberedBvecHost
            [iCell * numDofsDst + renumberingMap[iDoF]] =
              flattenedArrayCellLocalProcIndexIdMap[iCell * numDofsDst + iDoF] *
              BVec * numWfnSpinors;
          flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRemHost
            [iCell * numDofsDst + renumberingMap[iDoF]] =
              flattenedArrayCellLocalProcIndexIdMap[iCell * numDofsDst + iDoF] *
              (totalNumWaveFunctions % BVec) * numWfnSpinors;
        }
    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvec,
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRem;
    flattenedArrayCellLocalProcIndexIdMapRenumberedBvec.resize(
      flattenedArrayCellLocalProcIndexIdMap.size(), 0);
    flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRem.resize(
      flattenedArrayCellLocalProcIndexIdMap.size(), 0);
    flattenedArrayCellLocalProcIndexIdMapRenumberedBvec.copyFrom(
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvecHost);
    flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRem.copyFrom(
      flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRemHost);

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
           ++spinIndex)
        {
          wfcQuadPointData.setValue(zero);
          for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
               jvec += BVec)
            {
              const unsigned int currentBlockSize =
                std::min(BVec, totalNumWaveFunctions - jvec);
              flattenedArrayBlock = &(basisOperationsPtrSrc->getMultiVector(
                currentBlockSize * numWfnSpinors, 0));
              if (currentBlockSize == BVec)
                flattenedArrayBlockDst = &flattenedArrayBlockDstBVec;
              else
                flattenedArrayBlockDst = &flattenedArrayBlockDstBVecRem;
              if ((jvec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    currentBlockSize,
                    totalNumWaveFunctions,
                    numLocalDofs * numWfnSpinors,
                    jvec,
                    Psi.data() + numLocalDofs * numWfnSpinors *
                                   totalNumWaveFunctions *
                                   (numSpinComponents * kPoint + spinIndex),
                    flattenedArrayBlock->data());
                  basisOperationsPtrSrc->reinit(currentBlockSize *
                                                  numWfnSpinors,
                                                cellsBlockSize,
                                                quadratureIndex,
                                                false);


                  flattenedArrayBlock->updateGhostValues();
                  basisOperationsPtrSrc->distribute(*(flattenedArrayBlock));

                  for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                    {
                      const unsigned int currentCellsBlockSize =
                        (iblock == numCellBlocks) ? remCellBlockSize :
                                                    cellsBlockSize;
                      if (currentCellsBlockSize > 0)
                        {
                          const unsigned int startingCellId =
                            iblock * cellsBlockSize;

                          basisOperationsPtrSrc->interpolateKernel(
                            *(flattenedArrayBlock),
                            wfcQuadPointData.data(),
                            NULL,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));
                          BLASWrapperPtr->stridedCopyFromBlock(
                            currentBlockSize * numWfnSpinors,
                            numDofsDst * currentCellsBlockSize,
                            wfcQuadPointData.data(),
                            flattenedArrayBlockDst->data(),
                            (currentBlockSize == BVec ?
                               flattenedArrayCellLocalProcIndexIdMapRenumberedBvec
                                 .data() :
                               flattenedArrayCellLocalProcIndexIdMapRenumberedBvecRem
                                 .data()) +
                              startingCellId * numDofsDst);
                        } // non-trivial cell block check
                    }     // cells block loop
                  constraintInfoDst.set_zero(*flattenedArrayBlockDst);
                  flattenedArrayBlockDst->zeroOutGhosts();
                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    totalNumWaveFunctions,
                    currentBlockSize,
                    matrixFreeDataTemp.get_vector_partitioner()
                        ->locally_owned_size() *
                      numWfnSpinors,
                    jvec,
                    (*flattenedArrayBlockDst).data(),
                    PsiDst.data() + numLocalDofsDst * numWfnSpinors *
                                      totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex));
                }
            }
        }
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    std::swap(Psi, PsiDst);
    MPI_Barrier(mpiCommParent);
    interpolate_time = MPI_Wtime() - interpolate_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for wavefunction interpolation on CPU: "
                  << interpolate_time << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for wavefunction interpolation on Device: "
                  << interpolate_time << std::endl;
  }
#include "dft.inst.cc"
} // namespace dftfe
