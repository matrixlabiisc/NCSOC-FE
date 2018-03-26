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

/** @file meshGenerator.h
 *
 *  @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
 *
 *  @author Phani Motamarri, Sambit Das, Krishnendu Ghosh
 */

#ifndef meshGenerator_H_
#define meshGenerator_H_
#include "headers.h"

using namespace dealii;


class meshGeneratorClass
{

 public:
  /**
   * meshGeneratorClass constructor
   */
  meshGeneratorClass(const MPI_Comm &mpi_comm_replica);


  /**
   * meshGeneratorClass destructor
   */
  ~meshGeneratorClass();

  void generateSerialUnmovedAndParallelMovedUnmovedMesh
              (const std::vector<std::vector<double> > & atomLocations,
	       const std::vector<std::vector<double> > & imageAtomLocations,
	       const std::vector<std::vector<double> > & domainBoundingVectors);

  void generateParallelUnmovedPreviousMesh
              (const std::vector<std::vector<double> > & atomLocations,
	       const std::vector<std::vector<double> > & imageAtomLocations,
	       const std::vector<std::vector<double> > & domainBoundingVectors);

  const parallel::distributed::Triangulation<3> & getSerialMeshUnmoved();

  const parallel::distributed::Triangulation<3> & getParallelMeshMoved();

  const parallel::distributed::Triangulation<3> & getParallelMeshUnmoved();

  const parallel::distributed::Triangulation<3> & getParallelMeshUnmovedPrevious();

 private:

  void generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation, parallel::distributed::Triangulation<3>& serialTriangulation);


  void generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation);

  void refineSerialMesh(unsigned int n_cell, std::vector<double>& centroid, std::vector<int>& localRefineFlag, unsigned int n_global_cell, parallel::distributed::Triangulation<3>& serialTriangulation);
  //
  //data members
  //
  parallel::distributed::Triangulation<3> d_parallelTriangulationUnmoved;
  parallel::distributed::Triangulation<3> d_parallelTriangulationUnmovedPrevious;
  parallel::distributed::Triangulation<3> d_parallelTriangulationMoved;

  parallel::distributed::Triangulation<3> d_serialTriangulationUnmoved;

  std::vector<std::vector<double> > d_atomPositions;
  std::vector<std::vector<double> > d_imageAtomPositions;
  std::vector<std::vector<double> > d_domainBoundingVectors;

  //
  //parallel objects
  //
  const MPI_Comm mpi_communicator;
  const unsigned int this_mpi_process;
  const unsigned int n_mpi_processes;
  dealii::ConditionalOStream   pcout;

  //
  //compute-time logger
  //
  TimerOutput computing_timer;

};

#endif
