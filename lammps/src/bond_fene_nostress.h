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

#ifdef BOND_CLASS
BondStyle(fene/nostress, BondFENENOSTRESS);
#else

#ifndef LMP_BOND_FENE_NO_STRESS_H
#define LMP_BOND_FENE_NO_STRESS_H

#include "bond.h"

namespace LAMMPS_NS {

class BondFENENOSTRESS : public Bond {
 public:
  BondFENENOSTRESS(class LAMMPS *);
  ~BondFENENOSTRESS() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void born_matrix(int, double, int, int, double &, double &) override;
  void *extract(const char *, int &) override;

 protected:
  double *k, *r0, *epsilon, *sigma;

  virtual void allocate();
};

}

#endif
#endif
