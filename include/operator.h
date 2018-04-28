//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// --------------------------------------------------------------------------------------
//
// @author Phani Motamarri (2018)
//
#ifndef operatorDFTClass_h
#define operatorDFTClass_h

#include <vector>

#include "headers.h"

typedef dealii::parallel::distributed::Vector<double> vectorType;
namespace dftfe{
  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   */
  class operatorDFTClass {

    //
    // types
    //
  public:
    //
    // methods
    //
  public:

    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTClass() = 0;


    /**
     * @brief initialize operatorClass
     *
     */
    virtual void init() = 0;


    /**
     * @brief compute M matrix
     *
     * @return diagonal M matrix
     */
    virtual void computeMassVector() = 0;


    /**
     * @brief Compute operator times vector or operator times bunch of vectors
     *
     * @param X Vector of Vectors containing current values of X
     * @param Y Vector of Vectors containing operator times vectors product
     */
    virtual void HX(std::vector<vectorType> & x,
		    std::vector<vectorType> & y) = 0;


    /**
     * @brief Compute projection of the operator into orthogonal basis
     *
     * @param X given orthogonal basis vectors
     * @return ProjMatrix projected small matrix 
     */
#ifdef ENABLE_PERIODIC_BC
    virtual void XtHX(std::vector<vectorType> & X,
		      std::vector<std::complex<double> > & ProjHam) = 0;
#else
    virtual void XtHX(std::vector<vectorType> & X,
		      std::vector<double> & ProjHam) = 0;
#endif


    /**
     * @brief Get local dof indices real
     *
     * @return pointer to local dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalDofIndicesReal() const;

    /**
     * @brief Get local dof indices imag
     *
     * @return pointer to local dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalDofIndicesImag() const;

    /**
     * @brief Get local proc dof indices real
     *
     * @return pointer to local proc dof indices real
     */
    const std::vector<dealii::types::global_dof_index> * getLocalProcDofIndicesReal() const;


    /**
     * @brief Get local proc dof indices imag
     *
     * @return pointer to local proc dof indices imag
     */
    const std::vector<dealii::types::global_dof_index> * getLocalProcDofIndicesImag() const;

    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    const dealii::ConstraintMatrix * getConstraintMatrixEigen() const;


    /**
     * @brief Get relevant mpi communicator
     *
     * @return mpi communicator
     */
    const MPI_Comm & getMPICommunicator() const;
  

  protected:
    
    /**
     * @brief default Constructor.
     */
    operatorDFTClass();


    /**
     * @brief Constructor.
     */
    operatorDFTClass(const MPI_Comm & mpi_comm_replica,
		     const std::vector<dealii::types::global_dof_index> & localDofIndicesReal,
		     const std::vector<dealii::types::global_dof_index> & localDofIndicesImag,
		     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
		     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
		     const dealii::ConstraintMatrix  & constraintMatrixEigen);

  protected:


    //
    //global indices of degrees of freedom in the current processor which correspond to component-1 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localDofIndicesReal;

    //
    //global indices of degrees of freedom in the current processor which correspond to component-2 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localDofIndicesImag;

    //
    //local indices degrees of freedom in the current processor  which correspond to component-1 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localProcDofIndicesReal;

    //
    //local indices degrees of freedom in the current processor  which correspond to component-2 of 2-component dealii array
    //
    const std::vector<dealii::types::global_dof_index> * d_localProcDofIndicesImag;

    //
    //constraint matrix used in eigen solve
    //
    const dealii::ConstraintMatrix  * d_constraintMatrixEigen;

    //
    //mpi communicator
    //
    MPI_Comm                          d_mpi_communicator;

  };

}
#endif
