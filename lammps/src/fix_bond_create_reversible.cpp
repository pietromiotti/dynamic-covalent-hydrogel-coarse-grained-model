/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_bond_create_reversible.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"
#include <iostream>
#include "stdio.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define BIG 1.0e20
#define DELTA 16

FixBondCreateReversible::FixBondCreateReversible(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  bondcount(nullptr), partner(nullptr), finalpartner(nullptr), distsq(nullptr),
  probability(nullptr), created(nullptr), copy(nullptr), random(nullptr), list(nullptr)
{
  if (narg < 8) error->all(FLERR,"Illegal fix bond/create command");

  MPI_Comm_rank(world,&me);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/create command");

  dynamic_group_allow = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  active = true;
  iatomtype = utils::inumeric(FLERR,arg[4],false,lmp);
  jatomtype = utils::inumeric(FLERR,arg[5],false,lmp);
  double cutoff = utils::numeric(FLERR,arg[6],false,lmp);
  btype = utils::inumeric(FLERR,arg[7],false,lmp);

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix bond/create command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/create command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/create command");

  cutsq = cutoff*cutoff;

  imaxbond = 0;
  inewtype = iatomtype;
  jmaxbond = 0;
  jnewtype = jatomtype;
  fraction = 1.0;
  int seed = 12345;
  atype = dtype = itype = 0;

  constrainflag = 0;
  constrainpass = 0;
  amin = 0;
  amax = 180;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"iparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create command");
      imaxbond = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      inewtype = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (imaxbond < 0) error->all(FLERR,"Illegal fix bond/create command");
      if (inewtype < 1 || inewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"jparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create command");
      jmaxbond = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      jnewtype = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (jmaxbond < 0) error->all(FLERR,"Illegal fix bond/create command");
      if (jnewtype < 1 || jnewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create command");
      fraction = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      seed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/create command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"atype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      atype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (atype < 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"dtype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      dtype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (dtype < 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"itype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create command");
      itype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (itype < 0) error->all(FLERR,"Illegal fix bond/create command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"aconstrain") == 0 &&
        strcmp(style,"bond/create/angle") == 0) {
      if (iarg+3 > narg)
          error->all(FLERR,"Illegal fix bond/create/angle command");
      amin = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      amax = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (amin  >= amax)
        error->all(FLERR,"Illegal fix bond/create/angle command");
      if (amin < 0 || amin > 180)
        error->all(FLERR,"Illegal fix bond/create/angle command");
      if (amax < 0 || amax > 180)
        error->all(FLERR,"Illegal fix bond/create/angle command");
      amin = (MY_PI/180.0) * amin;
      amax = (MY_PI/180.0) * amax;
      constrainflag = 1;
      iarg += 3;
    } else error->all(FLERR,"Illegal fix bond/create command");
  }

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/create with non-molecular systems");
  if (iatomtype == jatomtype &&
      ((imaxbond != jmaxbond) || (inewtype != jnewtype)))
    error->all(FLERR,
               "Inconsistent iparam/jparam values in fix bond/create command");

  random = new RanMars(lmp,seed + me);

  bondcount = nullptr;
  FixBondCreateReversible::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  countflag = 0;

  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 2;

  nmax = 0;
  partner = finalpartner = nullptr;
  distsq = nullptr;

  maxcreate = 0;
  created = nullptr;

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  createcount = 0;
  createcounttotal = 0;
}

FixBondCreateReversible::~FixBondCreateReversible()
{

  atom->delete_callback(id,Atom::GROW);

  delete random;

  memory->destroy(bondcount);
  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(created);
  delete [] copy;
}

int FixBondCreateReversible::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

void FixBondCreateReversible::init()
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;

  if (force->pair == nullptr || cutsq > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/create cutoff is longer than pairwise cutoff");

  int count = 0;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style,"bond/create") == 0) count++;
    if (strcmp(modify->fix[i]->style,"bond/break") == 0) count++;
  }
  if (count > 1 && me == 0)
    error->warning(FLERR,"Fix bond/create is used multiple times "
                   " or with fix bond/break - may not work as expected");

  if (atype && force->angle) {
    angleflag = 1;
    if (atype > atom->nangletypes)
      error->all(FLERR,"Fix bond/create angle type is invalid");
  } else angleflag = 0;

  if (dtype && force->dihedral) {
    dihedralflag = 1;
    if (dtype > atom->ndihedraltypes)
      error->all(FLERR,"Fix bond/create dihedral type is invalid");
  } else dihedralflag = 0;

  if (itype && force->improper) {
    improperflag = 1;
    if (itype > atom->nimpropertypes)
      error->all(FLERR,"Fix bond/create improper type is invalid");
  } else improperflag = 0;

  if (force->improper) {
    if (force->improper_match("class2") || force->improper_match("ring"))
      error->all(FLERR,"Cannot yet use fix bond/create with this "
                 "improper style");
  }

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);

  lastcheck = -1;
}

void FixBondCreateReversible::init_list(int , NeighList *ptr)
{
  list = ptr;
}

void FixBondCreateReversible::initial_integrate(int )
{
  if (update->ntimestep % nevery) return;
  countflag = 0;
  setup(0);
}

void FixBondCreateReversible::setup(int )
{
  int i,j,m;

  if (countflag) return;
  countflag = 1;

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,"Fix bond/create needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }

  commflag = 1;
  if (newton_bond) comm->reverse_comm(this,1);
}

void FixBondCreateReversible::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if ((update->ntimestep % nevery)) {
	  return;
  }

  comm->forward_comm();

  commflag = 1;
  comm->forward_comm(this,1);

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/create:partner");
    memory->create(finalpartner,nmax,"bond/create:finalpartner");
    memory->create(distsq,nmax,"bond/create:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    distsq[i] = BIG;
  }

  double **x = atom->x;
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;
  int *type = atom->type;

  if (constrainflag) {

    commflag = 3;
    comm->forward_comm(this);
  }

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];

      possible = 0;
      if (itype == iatomtype && jtype == jatomtype) {
        if ((imaxbond == 0 || bondcount[i] < imaxbond) &&
            (jmaxbond == 0 || bondcount[j] < jmaxbond))
          possible = 1;
      } else if (itype == jatomtype && jtype == iatomtype) {
        if ((jmaxbond == 0 || bondcount[i] < jmaxbond) &&
            (imaxbond == 0 || bondcount[j] < imaxbond))
          possible = 1;
      }
      if (!possible) continue;

      for (k = 0; k < nspecial[i][0]; k++)
        if (special[i][k] == tag[j]) possible = 0;
      if (!possible) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) continue;

      if (constrainflag) {
        constrainpass = constrain(i,j,amin,amax);
        if (!constrainpass) continue;
      }

      if (rsq < distsq[i]) {
        partner[i] = tag[j];
        distsq[i] = rsq;
      }
      if (rsq < distsq[j]) {
        partner[j] = tag[i];
        distsq[j] = rsq;
      }

    }
  }

  commflag = 2;
  if (force->newton_pair) comm->reverse_comm(this);

  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 2;
  comm->forward_comm(this,2);

  int **bond_type = atom->bond_type;
  int newton_bond = force->newton_bond;

  ncreate = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    if (fraction < 1.0) {
      if (tag[i] < tag[j]) {
        if (probability[i] >= fraction) continue;
      } else {
        if (probability[j] >= fraction) continue;
      }
    }

    if (!newton_bond || tag[i] < tag[j]) {
      if (num_bond[i] == atom->bond_per_atom)
        error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create");
      bond_type[i][num_bond[i]] = btype;
      bond_atom[i][num_bond[i]] = tag[j];
      num_bond[i]++;
    }

    slist = special[i];
    n1 = nspecial[i][0];
    n2 = nspecial[i][1];
    n3 = nspecial[i][2];
    for (m = n1; m < n3; m++)
      if (slist[m] == tag[j]) break;
    if (m < n3) {
      for (n = m; n < n3-1; n++) slist[n] = slist[n+1];
      n3--;
      if (m < n2) n2--;
    }
    if (n3 == atom->maxspecial)
      error->one(FLERR,
                 "New bond exceeded special list size in fix bond/create");
    for (m = n3; m > n1; m--) slist[m] = slist[m-1];
    slist[n1] = tag[j];
    nspecial[i][0] = n1+1;
    nspecial[i][1] = n2+1;
    nspecial[i][2] = n3+1;

    bondcount[i]++;
    if (type[i] == iatomtype) {
      if (bondcount[i] == imaxbond) type[i] = inewtype;
    } else {
      if (bondcount[i] == jmaxbond) type[i] = jnewtype;
    }

    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) ncreate++;
  }

  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  if (createcount) next_reneighbor = update->ntimestep;
  if (!createcount) return;

  commflag = 3;
  comm->forward_comm(this);

  ncreate = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (ncreate == maxcreate) {
        maxcreate += DELTA;
        memory->grow(created,maxcreate,2,"bond/create:created");
      }
      created[ncreate][0] = tag[i];
      created[ncreate][1] = finalpartner[i];
      ncreate++;
    }
  }

  update_topology();
}

void FixBondCreateReversible::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Fix bond/create needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

void FixBondCreateReversible::update_topology()
{
  int i,j,k,n,influence,influenced;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;
  overflow = 0;

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < ncreate; j++) {
      id1 = created[j][0];
      id2 = created[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][1];
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) {
            influence = 1;
            break;
          }
      }
      if (!influence) continue;
      influenced = 1;
    }

    if (influenced) {
      rebuild_special_one(i);
      if (angleflag) create_angles(i);
      if (dihedralflag) create_dihedrals(i);
      if (improperflag) create_impropers(i);
    }
  }

  int overflowall;
  MPI_Allreduce(&overflow,&overflowall,1,MPI_INT,MPI_SUM,world);
  if (overflowall) error->all(FLERR,"Fix bond/create induced too many "
                              "angles/dihedrals/impropers per atom");

  int newton_bond = force->newton_bond;

  int all;
  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    atom->nangles += all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->ndihedrals += all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->nimpropers += all;
  }
}

void FixBondCreateReversible::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

void FixBondCreateReversible::create_angles(int m)
{
  int i,j,n,i2local,n1,n2;
  tagint i1,i2,i3;
  tagint *s1list,*s2list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_angle = atom->num_angle[m];
  int *angle_type = atom->angle_type[m];
  tagint *angle_atom1 = atom->angle_atom1[m];
  tagint *angle_atom2 = atom->angle_atom2[m];
  tagint *angle_atom3 = atom->angle_atom3[m];

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];

      for (n = 0; n < ncreate; n++) {
        if (created[n][0] == i1 && created[n][1] == i2) break;
        if (created[n][0] == i2 && created[n][1] == i1) break;
        if (created[n][0] == i2 && created[n][1] == i3) break;
        if (created[n][0] == i3 && created[n][1] == i2) break;
      }
      if (n == ncreate) continue;

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nangles++;
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
  if (force->newton_bond) return;

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    i2local = atom->map(i2);
    if (i2local < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    s2list = special[i2local];
    n2 = nspecial[i2local][0];

    for (j = 0; j < n2; j++) {
      i3 = s2list[j];
      if (i3 == i1) continue;

      for (n = 0; n < ncreate; n++) {
        if (created[n][0] == i1 && created[n][1] == i2) break;
        if (created[n][0] == i2 && created[n][1] == i1) break;
        if (created[n][0] == i2 && created[n][1] == i3) break;
        if (created[n][0] == i3 && created[n][1] == i2) break;
      }
      if (n == ncreate) continue;

      if (num_angle < atom->angle_per_atom) {
        angle_type[num_angle] = atype;
        angle_atom1[num_angle] = i1;
        angle_atom2[num_angle] = i2;
        angle_atom3[num_angle] = i3;
        num_angle++;
        nangles++;
      } else overflow = 1;
    }
  }

  atom->num_angle[m] = num_angle;
}

void FixBondCreateReversible::create_dihedrals(int m)
{
  int i,j,k,n,i1local,i2local,i3local,n1,n2,n3;
  tagint i1,i2,i3,i4;
  tagint *s1list,*s2list,*s3list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_dihedral = atom->num_dihedral[m];
  int *dihedral_type = atom->dihedral_type[m];
  tagint *dihedral_atom1 = atom->dihedral_atom1[m];
  tagint *dihedral_atom2 = atom->dihedral_atom2[m];
  tagint *dihedral_atom3 = atom->dihedral_atom3[m];
  tagint *dihedral_atom4 = atom->dihedral_atom4[m];

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];

    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];
      if (force->newton_bond && i2 > i3) continue;
      i3local = atom->map(i3);
      if (i3local < 0)
        error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
      s3list = special[i3local];
      n3 = nspecial[i3local][0];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i3 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i3) break;
        }
        if (n < ncreate) {
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i1;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i3;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1;
        }
      }
    }
  }

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    if (force->newton_bond && i2 > i1) continue;
    i1local = atom->map(i1);
    if (i1local < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    s3list = special[i1local];
    n3 = nspecial[i1local][0];

    for (j = i+1; j < n2; j++) {
      i3 = s2list[j];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n < ncreate) {
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i3;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i1;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1;
        }
      }
    }
  }

  atom->num_dihedral[m] = num_dihedral;
  if (force->newton_bond) return;

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    i2local = atom->map(i2);
    if (i2local < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    s2list = special[i2local];
    n2 = nspecial[i2local][0];

    for (j = 0; j < n2; j++) {
      i3 = s2list[j];
      if (i3 == i1) continue;
      i3local = atom->map(i3);
      if (i3local < 0)
        error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
      s3list = special[i3local];
      n3 = nspecial[i3local][0];

      for (k = 0; k < n3; k++) {
        i4 = s3list[k];
        if (i4 == i1 || i4 == i2 || i4 == i3) continue;

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i2 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i2) break;
          if (created[n][0] == i3 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i3) break;
        }
        if (n < ncreate) {
          if (num_dihedral < atom->dihedral_per_atom) {
            dihedral_type[num_dihedral] = dtype;
            dihedral_atom1[num_dihedral] = i1;
            dihedral_atom2[num_dihedral] = i2;
            dihedral_atom3[num_dihedral] = i3;
            dihedral_atom4[num_dihedral] = i4;
            num_dihedral++;
            ndihedrals++;
          } else overflow = 1;
        }
      }
    }
  }
}

void FixBondCreateReversible::create_impropers(int m)
{
  int i,j,k,n,i1local,n1,n2;
  tagint i1,i2,i3,i4;
  tagint *s1list,*s2list;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int num_improper = atom->num_improper[m];
  int *improper_type = atom->improper_type[m];
  tagint *improper_atom1 = atom->improper_atom1[m];
  tagint *improper_atom2 = atom->improper_atom2[m];
  tagint *improper_atom3 = atom->improper_atom3[m];
  tagint *improper_atom4 = atom->improper_atom4[m];

  i1 = tag[m];
  n1 = nspecial[m][0];
  s1list = special[m];

  for (i = 0; i < n1; i++) {
    i2 = s1list[i];
    for (j = i+1; j < n1; j++) {
      i3 = s1list[j];
      for (k = j+1; k < n1; k++) {
        i4 = s1list[k];

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n == ncreate) continue;

        if (num_improper < atom->improper_per_atom) {
          improper_type[num_improper] = itype;
          improper_atom1[num_improper] = i1;
          improper_atom2[num_improper] = i2;
          improper_atom3[num_improper] = i3;
          improper_atom4[num_improper] = i4;
          num_improper++;
          nimpropers++;
        } else overflow = 1;
      }
    }
  }

  atom->num_improper[m] = num_improper;
  if (force->newton_bond) return;

  i2 = tag[m];
  n2 = nspecial[m][0];
  s2list = special[m];

  for (i = 0; i < n2; i++) {
    i1 = s2list[i];
    i1local = atom->map(i1);
    if (i1local < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    s1list = special[i1local];
    n1 = nspecial[i1local][0];

    for (j = 0; j < n1; j++) {
      i3 = s1list[j];
      if (i3 == i1 || i3 == i2) continue;

      for (k = j+1; k < n1; k++) {
        i4 = s1list[k];
        if (i4 == i1 || i4 == i2) continue;

        for (n = 0; n < ncreate; n++) {
          if (created[n][0] == i1 && created[n][1] == i2) break;
          if (created[n][0] == i2 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i3) break;
          if (created[n][0] == i3 && created[n][1] == i1) break;
          if (created[n][0] == i1 && created[n][1] == i4) break;
          if (created[n][0] == i4 && created[n][1] == i1) break;
        }
        if (n < ncreate) {
          if (num_improper < atom->improper_per_atom) {
            improper_type[num_improper] = itype;
            improper_atom1[num_improper] = i1;
            improper_atom2[num_improper] = i2;
            improper_atom3[num_improper] = i3;
            improper_atom4[num_improper] = i4;
            num_improper++;
            nimpropers++;
          } else overflow = 1;
        }
      }
    }
  }
}

int FixBondCreateReversible::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

void FixBondCreateReversible::post_integrate_respa(int ilevel, int )
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

int FixBondCreateReversible::pack_forward_comm(int n, int *list, double *buf,
                                     int , int * )
{
  int i,j,k,m,ns;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bondcount[j]).d;
    }
    return m;
  }

  if (commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = probability[j];
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

void FixBondCreateReversible::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      bondcount[i] = (int) ubuf(buf[m++]).i;

  } else if (commflag == 2) {
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
      probability[i] = buf[m++];
    }

  } else {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

int FixBondCreateReversible::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      buf[m++] = ubuf(bondcount[i]).d;
    return m;
  }

  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = distsq[i];
  }
  return m;
}

void FixBondCreateReversible::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += (int) ubuf(buf[m++]).i;
    }

  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] < distsq[j]) {
        partner[j] = (tagint) ubuf(buf[m++]).i;
        distsq[j] = buf[m++];
      } else m += 2;
    }
  }
}

void FixBondCreateReversible::grow_arrays(int nmax)
{
  memory->grow(bondcount,nmax,"bond/create:bondcount");
}

void FixBondCreateReversible::copy_arrays(int i, int j, int )
{
  bondcount[j] = bondcount[i];
}

int FixBondCreateReversible::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

int FixBondCreateReversible::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

double FixBondCreateReversible::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  return (double) createcounttotal;
}

double FixBondCreateReversible::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax * sizeof(int);
  bytes += 2*nmax * sizeof(tagint);
  bytes += (double)nmax * sizeof(double);
  return bytes;
}
