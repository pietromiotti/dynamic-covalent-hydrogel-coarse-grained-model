/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
ComputeStyle(activebonds,ComputeActiveBonds);
#else

#ifndef LMP_COMPUTE_ACTIVE_BONDS_H
#define LMP_COMPUTE_ACTIVE_BONDS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeActiveBonds : public Compute {
 public:
  ComputeActiveBonds(class LAMMPS *, int, char **);
  ~ComputeActiveBonds() override;

  void init() override;
  void compute_vector() override;
};

}

#endif
#endif
