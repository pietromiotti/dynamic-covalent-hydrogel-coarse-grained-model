/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   ----------------------------------------------------------------------- */

#include "compute_active_bonds.h"

#include "atom.h"
#include "error.h"
#include "update.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

ComputeActiveBonds::ComputeActiveBonds(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute momentum command");

  vector_flag = 1;
  size_vector = 3;
  extvector = 1;
  vector = new double[size_vector];
}

ComputeActiveBonds::~ComputeActiveBonds() {
  delete[] vector;
}

void ComputeActiveBonds::init()
{
}

void ComputeActiveBonds::compute_vector()
{
  invoked_vector = update->ntimestep;

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double activebonds[3] = {0.0,0.0,0.0};

  for (int n = 0; n < nbondlist; n++) {
    int type = bondlist[n][2];
    if (type == 2) activebonds[0]++;
  }

  MPI_Allreduce(activebonds, vector, 1, MPI_DOUBLE, MPI_SUM, world);
}
