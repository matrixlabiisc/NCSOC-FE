// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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

namespace internalLowrankJacInv
{
  double
  relativeErrorEstimate(
    const std::deque<distributedCPUVec<double>> &fvcontainer,
    const distributedCPUVec<double> &            residualVec,
    const double                                 k0)
  {
    const unsigned int rank = fvcontainer.size();

    std::vector<double> mMat(rank * rank, 0.0);
    for (int j = 0; j < rank; j++)
      for (int i = 0; i < rank; i++)
        mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

    dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

    distributedCPUVec<double> k0ResidualVec, approximationErrorVec;
    k0ResidualVec.reinit(residualVec);
    approximationErrorVec.reinit(residualVec);
    for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
      {
        k0ResidualVec.local_element(idof) =
          residualVec.local_element(idof) * k0;
        approximationErrorVec.local_element(idof) =
          k0ResidualVec.local_element(idof);
      }

    std::vector<double> innerProducts(rank, 0.0);
    for (unsigned int i = 0; i < rank; i++)
      innerProducts[i] = fvcontainer[i] * k0ResidualVec;


    for (unsigned int i = 0; i < rank; i++)
      {
        double temp = 0.0;
        for (unsigned int j = 0; j < rank; j++)
          temp += mMat[j * rank + i] * innerProducts[j];

        for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
          approximationErrorVec.local_element(idof) -=
            fvcontainer[i].local_element(idof) * temp;
      }

    return (approximationErrorVec.l2_norm() / k0ResidualVec.l2_norm());
  }

  void
  lowrankKernelApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                     const std::deque<distributedCPUVec<double>> &vcontainer,
                     const distributedCPUVec<double> &            x,
                     const double                                 k0,
                     distributedCPUVec<double> &                  y)
  {
    const unsigned int rank = fvcontainer.size();

    std::vector<double> mMat(rank * rank, 0.0);
    for (int j = 0; j < rank; j++)
      for (int i = 0; i < rank; i++)
        mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

    dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

    // FIXME: separete k0x vector is not required, use the y vector
    distributedCPUVec<double> k0x;
    k0x.reinit(x);
    for (unsigned int idof = 0; idof < x.local_size(); idof++)
      k0x.local_element(idof) = x.local_element(idof) * k0;

    std::vector<double> innerProducts(rank, 0.0);
    for (unsigned int i = 0; i < rank; i++)
      innerProducts[i] = fvcontainer[i] * k0x;

    y = 0;

    for (unsigned int i = 0; i < rank; i++)
      {
        double temp = 0.0;
        // FIXME: exploit symmetry of mMat
        for (unsigned int j = 0; j < rank; j++)
          temp += mMat[j * rank + i] * innerProducts[j];

        for (unsigned int idof = 0; idof < y.local_size(); idof++)
          y.local_element(idof) += vcontainer[i].local_element(idof) * temp;
      }
  }


  void
  lowrankJacInvApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                     const std::deque<distributedCPUVec<double>> &vcontainer,
                     const distributedCPUVec<double> &            x,
                     distributedCPUVec<double> &                  y)
  {
    const unsigned int rank = fvcontainer.size();

    std::vector<double> mMat(rank * rank, 0.0);
    for (int j = 0; j < rank; j++)
      for (int i = 0; i < rank; i++)
        mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

    dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

    std::vector<double> innerProducts(rank, 0.0);
    for (unsigned int i = 0; i < rank; i++)
      innerProducts[i] = fvcontainer[i] * x;

    y = 0;

    for (unsigned int i = 0; i < rank; i++)
      {
        double temp = 0.0;
        for (unsigned int j = 0; j < rank; j++)
          temp += mMat[j * rank + i] * innerProducts[j];

        for (unsigned int idof = 0; idof < y.local_size(); idof++)
          y.local_element(idof) += vcontainer[i].local_element(idof) * temp;
      }
  }


  void
  lowrankJacApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                  const std::deque<distributedCPUVec<double>> &vcontainer,
                  const distributedCPUVec<double> &            x,
                  distributedCPUVec<double> &                  y)
  {
    const unsigned int rank = fvcontainer.size();


    std::vector<double> innerProducts(rank, 0.0);
    for (unsigned int i = 0; i < rank; i++)
      innerProducts[i] = vcontainer[i] * x;

    y = 0;
    for (unsigned int i = 0; i < rank; i++)
      for (unsigned int idof = 0; idof < y.local_size(); idof++)
        y.local_element(idof) +=
          fvcontainer[i].local_element(idof) * innerProducts[i];
  }



  double
  estimateLargestEigenvalueMagJacLowrankPower(
    const std::deque<distributedCPUVec<double>> &lowrankFvcontainer,
    const std::deque<distributedCPUVec<double>> &lowrankVcontainer,
    const distributedCPUVec<double> &            x,
    const dealii::AffineConstraints<double> &    constraintsRhoNodal)
  {
    const double tol = 1.0e-6;

    double lambdaOld     = 0.0;
    double lambdaNew     = 0.0;
    double diffLambdaAbs = 1e+6;
    //
    // generate random vector v
    //
    distributedCPUVec<double> vVector, fVector;
    vVector.reinit(x);
    fVector.reinit(x);

    vVector = 0.0, fVector = 0.0;
    // std::srand(this_mpi_process);
    const unsigned int local_size = vVector.local_size();

    // for (unsigned int i = 0; i < local_size; i++)
    //  vVector.local_element(i) = x.local_element(i);

    for (unsigned int i = 0; i < local_size; i++)
      vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

    constraintsRhoNodal.set_zero(vVector);

    vVector.update_ghost_values();

    //
    // evaluate l2 norm
    //
    vVector /= vVector.l2_norm();
    vVector.update_ghost_values();
    int iter = 0;
    while (diffLambdaAbs > tol)
      {
        fVector = 0;
        lowrankJacApply(lowrankFvcontainer,
                        lowrankVcontainer,
                        vVector,
                        fVector);
        lambdaOld = lambdaNew;
        lambdaNew = (vVector * fVector) / (vVector * vVector);

        vVector = fVector;
        vVector /= vVector.l2_norm();
        vVector.update_ghost_values();
        diffLambdaAbs = std::abs(lambdaNew - lambdaOld);
        iter++;
      }

    // std::cout << " Power iterations iter: "<< iter
    //            << std::endl;

    return std::abs(lambdaNew);
  }

  double
  estimateLargestEigenvalueMagJacInvLowrankPower(
    const std::deque<distributedCPUVec<double>> &lowrankFvcontainer,
    const std::deque<distributedCPUVec<double>> &lowrankVcontainer,
    const distributedCPUVec<double> &            x,
    const dealii::AffineConstraints<double> &    constraintsRhoNodal)
  {
    const double tol = 1.0e-6;

    double lambdaOld     = 0.0;
    double lambdaNew     = 0.0;
    double diffLambdaAbs = 1e+6;
    //
    // generate random vector v
    //
    distributedCPUVec<double> vVector, fVector;
    vVector.reinit(x);
    fVector.reinit(x);

    vVector = 0.0, fVector = 0.0;
    // std::srand(this_mpi_process);
    const unsigned int local_size = vVector.local_size();

    // for (unsigned int i = 0; i < local_size; i++)
    //   vVector.local_element(i) = x.local_element(i);

    for (unsigned int i = 0; i < local_size; i++)
      vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

    constraintsRhoNodal.set_zero(vVector);

    vVector.update_ghost_values();

    //
    // evaluate l2 norm
    //
    vVector /= vVector.l2_norm();
    vVector.update_ghost_values();

    int iter = 0;
    while (diffLambdaAbs > tol)
      {
        fVector = 0;
        lowrankJacInvApply(lowrankFvcontainer,
                           lowrankVcontainer,
                           vVector,
                           fVector);
        lambdaOld = lambdaNew;
        lambdaNew = (vVector * fVector);

        vVector = fVector;
        vVector /= vVector.l2_norm();
        vVector.update_ghost_values();
        diffLambdaAbs = std::abs(lambdaNew - lambdaOld);
        iter++;
      }

    // std::cout << " Power iterations iter: "<< iter
    //            << std::endl;

    return std::abs(lambdaNew);
  }
} // namespace internalLowrankJacInv

template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::lowrankApproxScfJacobianInv(
  const unsigned int scfIter)
{
  int this_process;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
  MPI_Barrier(MPI_COMM_WORLD);
  double total_time = MPI_Wtime();

  double normValue = 0.0;

  distributedCPUVec<double> residualRho;
  residualRho.reinit(d_rhoInNodalValues);
  residualRho = 0.0;


  // compute residual = rhoOut - rhoIn
  residualRho.add(1.0, d_rhoOutNodalValues, -1.0, d_rhoInNodalValues);

  residualRho.update_ghost_values();

  // compute l2 norm of the field residual
  normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                             residualRho,
                             d_densityDofHandlerIndexElectro,
                             d_densityQuadratureIdElectro);

  const double k0 = 1.0;


  distributedCPUVec<double> kernelAction;
  distributedCPUVec<double> compvec;
  distributedCPUVec<double> dummy;
  kernelAction.reinit(residualRho);
  compvec.reinit(residualRho);
  double             charge;
  const unsigned int local_size = residualRho.local_size();

  const unsigned int maxRankCurrentSCF = 30;
  const unsigned int maxRankAccum      = 30;

  if (d_rankCurrent >= 1 &&
      dftParameters::methodSubTypeLRJI == "ACCUMULATED_ADAPTIVE")
    {
      const double relativeApproxError =
        internalLowrankJacInv::relativeErrorEstimate(d_fvcontainerVals,
                                                     residualRho,
                                                     k0);
      if (d_rankCurrent >= maxRankAccum ||
          (relativeApproxError > dftParameters::adaptiveRankRelTolLRJI *
                                   dftParameters::factorAdapAccumClearLRJI) ||
          relativeApproxError > d_relativeErrorJacInvApproxPrevScf)
        {
          if (dftParameters::verbosity >= 4)
            pcout
              << " Clearing accumulation as relative tolerance metric exceeded "
              << ", relative tolerance current scf: " << relativeApproxError
              << ", relative tolerance prev scf: "
              << d_relativeErrorJacInvApproxPrevScf << std::endl;
          d_vcontainerVals.clear();
          d_fvcontainerVals.clear();
          d_rankCurrent                      = 0;
          d_relativeErrorJacInvApproxPrevScf = 100.0;
        }
      else
        d_relativeErrorJacInvApproxPrevScf = relativeApproxError;
    }
  else
    {
      d_vcontainerVals.clear();
      d_fvcontainerVals.clear();
      d_rankCurrent = 0;
    }

  unsigned int       rankAddedInThisScf = 0;
  const unsigned int maxRankThisScf     = (scfIter < 2) ? 5 : maxRankCurrentSCF;
  while (rankAddedInThisScf < maxRankThisScf)
    {
      if (rankAddedInThisScf == 0)
        {
          d_vcontainerVals.push_back(residualRho);
          d_vcontainerVals[d_rankCurrent] *= k0;
        }
      else
        d_vcontainerVals.push_back(d_fvcontainerVals[d_rankCurrent - 1]);

      compvec = 0;
      for (int jrank = 0; jrank < d_rankCurrent; jrank++)
        {
          const double tTvj =
            d_vcontainerVals[d_rankCurrent] * d_vcontainerVals[jrank];
          compvec.add(tTvj, d_vcontainerVals[jrank]);
        }
      d_vcontainerVals[d_rankCurrent] -= compvec;

      d_vcontainerVals[d_rankCurrent] *=
        1.0 / d_vcontainerVals[d_rankCurrent].l2_norm();

      if (dftParameters::verbosity >= 4)
        pcout << " Vector norm of v:  "
              << d_vcontainerVals[d_rankCurrent].l2_norm()
              << ", for rank: " << d_rankCurrent + 1 << std::endl;

      d_fvcontainerVals.push_back(residualRho);
      d_fvcontainerVals[d_rankCurrent] = 0;

      d_vcontainerVals[d_rankCurrent].update_ghost_values();
      charge =
        totalCharge(d_matrixFreeDataPRefined, d_vcontainerVals[d_rankCurrent]);


      if (dftParameters::verbosity >= 4)
        pcout << "Integral v before scaling:  " << charge << std::endl;

      d_vcontainerVals[d_rankCurrent].add(-charge / d_domainVolume);

      d_vcontainerVals[d_rankCurrent].update_ghost_values();
      charge =
        totalCharge(d_matrixFreeDataPRefined, d_vcontainerVals[d_rankCurrent]);

      if (dftParameters::verbosity >= 4)
        pcout << "Integral v after scaling:  " << charge << std::endl;

      computeOutputDensityDirectionalDerivative(
        d_vcontainerVals[d_rankCurrent],
        dummy,
        dummy,
        d_fvcontainerVals[d_rankCurrent],
        dummy,
        dummy);

      d_fvcontainerVals[d_rankCurrent].update_ghost_values();
      charge =
        totalCharge(d_matrixFreeDataPRefined, d_fvcontainerVals[d_rankCurrent]);


      if (dftParameters::verbosity >= 4)
        pcout << "Integral fv before scaling:  " << charge << std::endl;

      d_fvcontainerVals[d_rankCurrent].add(-charge / d_domainVolume);

      d_fvcontainerVals[d_rankCurrent].update_ghost_values();
      charge =
        totalCharge(d_matrixFreeDataPRefined, d_fvcontainerVals[d_rankCurrent]);
      if (dftParameters::verbosity >= 4)
        pcout << "Integral fv after scaling:  " << charge << std::endl;

      if (dftParameters::verbosity >= 4)
        pcout
          << " Vector norm of response (delta rho_min[n+delta_lambda*v1]/ delta_lambda):  "
          << d_fvcontainerVals[d_rankCurrent].l2_norm()
          << " for kernel rank: " << d_rankCurrent + 1 << std::endl;

      d_fvcontainerVals[d_rankCurrent] -= d_vcontainerVals[d_rankCurrent];
      d_fvcontainerVals[d_rankCurrent] *= k0;
      d_rankCurrent++;
      rankAddedInThisScf++;

      if (dftParameters::methodSubTypeLRJI == "ADAPTIVE" ||
          dftParameters::methodSubTypeLRJI == "ACCUMULATED_ADAPTIVE")
        {
          const double relativeApproxError =
            internalLowrankJacInv::relativeErrorEstimate(d_fvcontainerVals,
                                                         residualRho,
                                                         k0);

          if (dftParameters::verbosity >= 4)
            pcout << " Relative approx error:  " << relativeApproxError
                  << " for kernel rank: " << d_rankCurrent << std::endl;

          if ((normValue < dftParameters::selfConsistentSolverTolerance) &&
              (dftParameters::estimateJacCondNoFinalSCFIter))
            {
              if (relativeApproxError < 1.0e-5)
                {
                  break;
                }
            }
          else
            {
              if (relativeApproxError < dftParameters::adaptiveRankRelTolLRJI)
                {
                  break;
                }
            }
        }
    }


  if (dftParameters::verbosity >= 4)
    pcout << " Net accumulated kernel rank:  " << d_rankCurrent
          << " Accumulated in this scf: " << rankAddedInThisScf << std::endl;

  internalLowrankJacInv::lowrankKernelApply(
    d_fvcontainerVals, d_vcontainerVals, residualRho, k0, kernelAction);

  if (normValue < dftParameters::selfConsistentSolverTolerance &&
      dftParameters::estimateJacCondNoFinalSCFIter)
    {
      const double maxAbsEigenValue =
        internalLowrankJacInv::estimateLargestEigenvalueMagJacLowrankPower(
          d_fvcontainerVals,
          d_vcontainerVals,
          residualRho,
          d_constraintsRhoNodal);
      const double minAbsEigenValue =
        1.0 /
        internalLowrankJacInv::estimateLargestEigenvalueMagJacInvLowrankPower(
          d_fvcontainerVals,
          d_vcontainerVals,
          residualRho,
          d_constraintsRhoNodal);
      pcout << " Maximum eigenvalue of low rank approx of Jacobian: "
            << maxAbsEigenValue << std::endl;
      pcout << " Minimum non-zero eigenvalue of low rank approx of Jacobian: "
            << minAbsEigenValue << std::endl;
      pcout << " Condition no of low rank approx of Jacobian: "
            << maxAbsEigenValue / minAbsEigenValue << std::endl;
    }

  // pcout << " Preconditioned simple mixing step " << std::endl;
  // preconditioned simple mixing step
  // Note for const=-1.0, it should be same as Newton step
  // For second scf iteration step (scIter==1), the rhoIn is from atomic
  // densities which casues robustness issues when used with a
  // higher mixingParameter value.
  // Suggested to use 0.1 for initial steps
  // as well as when normValue is greater than 2.0
  double const2 =
    (normValue > dftParameters::startingNormLRJILargeDamping || scfIter < 2) ?
      -0.1 :
      -dftParameters::mixingParameterLRJI;

  if (dftParameters::verbosity >= 4)
    pcout << " Preconditioned mixing step, mixing constant: " << const2
          << std::endl;

  d_rhoInNodalValues.add(const2, kernelAction);

  d_rhoInNodalValues.update_ghost_values();

  // interpolate nodal data to quadrature data
  interpolateRhoNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_densityDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    d_rhoInNodalValues,
    *rhoInValues,
    *gradRhoInValues,
    *gradRhoInValues,
    dftParameters::xcFamilyType == "GGA");

  // push the rhoIn to deque storing the history of nodal values
  d_rhoInNodalVals.push_back(d_rhoInNodalValues);

  MPI_Barrier(MPI_COMM_WORLD);
  total_time = MPI_Wtime() - total_time;

  if (this_process == 0 && dftParameters::verbosity >= 2)
    std::cout << "Time for low rank jac inv: " << total_time << std::endl;

  return normValue;
}
