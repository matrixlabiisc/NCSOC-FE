// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//
// @author Shiva Rudraraju, Phani Motamarri
//

#ifndef eigen_H_
#define eigen_H_
#include <headers.h>
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <operator.h>

namespace dftfe{

  typedef dealii::parallel::distributed::Vector<double> vectorType;
  template <unsigned int T> class dftClass;

  //
  //Define eigenClass class
  //
  template <unsigned int FEOrder>
    class eigenClass : public operatorDFTClass
    {
      template <unsigned int T>
	friend class dftClass;

      template <unsigned int T>
	friend class symmetryClass;

    public:
      eigenClass(dftClass<FEOrder>* _dftPtr, const MPI_Comm &mpi_comm_replica);

      /**
       * @brief Compute operator times vector or operator times bunch of vectors
       *
       * @param src Vector of Vectors containing current values of source array (non-const as
         we scale src and rescale src to avoid creation of temporary vectors)
       * @param dst Vector of Vectors containing operator times vectors product
       */
      void HX(std::vector<vectorType> &src, 
	      std::vector<vectorType> &dst);
      

      
     
#ifdef ENABLE_PERIODIC_BC
      /**
       * @brief Compute discretized operator matrix times multi-vectors
       *
       * @param src Vector containing current values of source array with multi-vector array stored 
       * in a flattened format with all the wavefunction value corresponding to a given node is stored 
       * contiguously (non-const as we scale src and rescale src to avoid creation of temporary vectors)
       * @param numberComponents Number of multi-fields(vectors)
       * @param macroCellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param cellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of local active cells
       * @param dst Vector containing operator times given multi-vectors product
       */
      void HX(dealii::parallel::distributed::Vector<std::complex<double> > & src,
	      const unsigned int numberComponents,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
	      bool scaleFlag,
	      const std::complex<double> scalar,
	      dealii::parallel::distributed::Vector<std::complex<double> > & dst);


      /**
       * @brief Compute projection of the operator into orthogonal basis
       *
       * @param src given orthogonal basis vectors 
       * @return ProjMatrix projected small matrix 
       */
      void XtHX(std::vector<vectorType> &src,
		std::vector<std::complex<double> > & ProjHam); 
#else
      
      /**
       * @brief Compute discretized operator matrix times multi-vectors
       *
       * @param src Vector containing current values of source array with multi-vector array stored 
       * in a flattened format with all the wavefunction value corresponding to a given node is stored 
       * contiguously (non-const as we scale src and rescale src to avoid creation of temporary vectors)
       * @param numberComponents Number of multi-fields(vectors)
       * @param macroCellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param cellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of local active cells
       * @param dst Vector containing operator times given multi-vectors product
       */
      void HX(dealii::parallel::distributed::Vector<double> & src,
	      const unsigned int numberComponents,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
	      const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
	      bool scaleFlag,
	      const double scalar,
	      dealii::parallel::distributed::Vector<double> & dst);

      /**
       * @brief Compute projection of the operator into orthogonal basis
       *
       * @param X given orthogonal basis vectors 
       * @return ProjMatrix projected small matrix 
       */
      void XtHX(std::vector<vectorType> &src,
		std::vector<double> & ProjHam);
#endif

     
       /**
       * @brief Computes effective potential involving local-density exchange-correlation functionals
       *
       * @param rhoValues electron-density
       * @param phi electrostatic potential arising both from electron-density and nuclear charge
       * @param phiExt electrostatic potential arising from nuclear charges
       * @param pseudoValues quadrature data of pseudopotential values
       */
      void computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);


      /**
       * @brief Computes effective potential involving local spin density exchange-correlation functionals
       *
       * @param rhoValues electron-density
       * @param phi electrostatic potential arising both from electron-density and nuclear charge
       * @param phiExt electrostatic potential arising from nuclear charges
       * @param spinIndex flag to toggle spin-up or spin-down
       * @param pseudoValues quadrature data of pseudopotential values
       */
      void computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				    const vectorType & phi,
				    const vectorType & phiExt,
				    unsigned int spinIndex,
				    const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

       /**
       * @brief Computes effective potential involving gradient density type exchange-correlation functionals
       *
       * @param rhoValues electron-density
       * @param gradRhoValues gradient of electron-density
       * @param phi electrostatic potential arising both from electron-density and nuclear charge
       * @param phiExt electrostatic potential arising from nuclear charges
       * @param pseudoValues quadrature data of pseudopotential values
       */
      void computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
		       const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
		       const vectorType & phi,
		       const vectorType & phiExt,
		       const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      
      /**
       * @brief Computes effective potential for gradient-spin density type exchange-correlation functionals
       *
       * @param rhoValues electron-density
       * @param gradRhoValues gradient of electron-density
       * @param phi electrostatic potential arising both from electron-density and nuclear charge
       * @param phiExt electrostatic potential arising from nuclear charges
       * @param spinIndex flag to toggle spin-up or spin-down
       * @param pseudoValues quadrature data of pseudopotential values
       */
      void computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				    const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				    const vectorType & phi,
				    const vectorType & phiExt,
				    const unsigned int spinIndex,
				    const std::map<dealii::CellId,std::vector<double> > & pseudoValues);

      
      /**
       * @brief sets the data member to appropriate kPoint Index 
       *
       * @param kPointIndex  k-point Index to set
       */
      void reinitkPointIndex(unsigned int & kPointIndex);

      //
      //initialize eigen class
      //
      void init ();
	    

      /**
       * @brief Computes diagonal mass matrix
       *
       * @param dofHandler dofHandler associated with the current mesh
       * @param constraintMatrix constraints to be used
       * @param sqrtMassVec output the value of square root of diagonal mass matrix 
       * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
       */
      void computeMassVector(const dealii::DoFHandler<3> & dofHandler,
	                     const dealii::ConstraintMatrix & constraintMatrix,
			     vectorType & sqrtMassVec,
			     vectorType & invSqrtMassVec);

      //precompute shapefunction gradient integral
      void preComputeShapeFunctionGradientIntegrals();

      //compute element Hamiltonian matrix
      void computeHamiltonianMatrix(unsigned int kPointIndex);





    private:
      /**
       * @brief implementation of matrix-free based matrix-vector product at cell-level
       * @param data matrix-free data
       * @param dst Vector of Vectors containing matrix times vectors product
       * @param src Vector of Vectors containing input vectors
       * @param cell_range range of cell-blocks
       */
      void computeLocalHamiltonianTimesXMF(const dealii::MatrixFree<3,double>  &data,
					   std::vector<vectorType>  &dst, 
					   const std::vector<vectorType>  &src,
					   const std::pair<unsigned int,unsigned int> &cell_range) const;

      /**
       * @brief implementation of  matrix-vector product for nonlocal Hamiltonian
       * @param src Vector of Vectors containing input vectors
       * @param dst Vector of Vectors containing matrix times vectors product
       */
      void computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
					    std::vector<vectorType>       &dst) const;  

  
     

      /**
       * @brief implementation of matrix-vector product using cell-level stiffness matrices
       * @param src Vector containing current values of source array with multi-vector array stored 
       * in a flattened format with all the wavefunction value corresponding to a given node is stored 
       * contiguously.
       * @param numberWaveFunctions Number of wavefunctions at a given node.
       * @param flattenedArrayCellLocalProcIndexIdMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param dst Vector containing matrix times given multi-vectors product
       */

#ifdef ENABLE_PERIODIC_BC
      //finite-element cell level stiffness matrix
      std::vector<std::vector<std::complex<double> > > d_cellHamiltonianMatrix;
      
      void computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
					 const int numberWaveFunctions,
					 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					 dealii::parallel::distributed::Vector<std::complex<double> > & dst) const;


#else
      //finite-element cell level stiffness matrix
      std::vector<std::vector<double> > d_cellHamiltonianMatrix;

      void computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
					 const int numberWaveFunctions,
					 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					 dealii::parallel::distributed::Vector<double> & dst) const;

#endif

      /**
       * @brief implementation of non-local Hamiltonian matrix-vector product using non-local discretized projectors at cell-level
       * @param src Vector containing current values of source array with multi-vector array stored 
       * in a flattened format with all the wavefunction value corresponding to a given node is stored 
       * contiguously.
       * @param numberWaveFunctions Number of wavefunctions at a given node.
       * @param flattenedArrayCellLocalProcIndexIdMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
       * @param dst Vector containing matrix times given multi-vectors product
       */

#ifdef ENABLE_PERIODIC_BC
      void computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
					    const int numberWaveFunctions,
					    const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					    dealii::parallel::distributed::Vector<std::complex<double> > & dst) const;

#else
      void computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
					    const int numberWaveFunctions,
					    const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
					    dealii::parallel::distributed::Vector<double> & dst) const;
#endif      

      //pointer to dft class
      dftClass<FEOrder>* dftPtr;


      //data structures
      vectorType d_invSqrtMassVector,d_sqrtMassVector;

      dealii::Table<2, dealii::VectorizedArray<double> > vEff;
      dealii::Table<3, dealii::VectorizedArray<double> > derExcWithSigmaTimesGradRho;


      //precomputed data for the Hamiltonian matrix
      std::vector<std::vector<dealii::VectorizedArray<double> > > d_cellShapeFunctionGradientIntegral;
      std::vector<double> d_shapeFunctionValue;
      dealii::Table<4, dealii::VectorizedArray<double> > d_cellShapeFunctionGradientValue;


      //
      //access to matrix-free cell data
      //
      const int d_numberNodesPerElement,d_numberMacroCells;
      std::vector<unsigned int> d_macroCellSubCellMap;

      //parallel objects
      const MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;
      dealii::ConditionalOStream   pcout;

      //compute-time logger
      dealii::TimerOutput computing_timer;

      //mutex thread for managing multi-thread writing to XHXvalue
      mutable dealii::Threads::Mutex  assembler_lock;

      //d_kpoint index for which Hamiltonian is computed
      unsigned int d_kPointIndex;

    };


}
#endif
