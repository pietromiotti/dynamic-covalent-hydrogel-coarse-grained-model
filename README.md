# dynamic-covalent-hydrogel-coarse-grained-model

Code and data for P. Miotti, L. Cousin, M. W. Tibbitt, I. V. Pivkin, *Mesoscopic Modeling of Dynamic Tetra-PEG Hydrogel Networks*.

Based on LAMMPS (stable 17 Apr 2024) 


## Run

```sh
cd lammps
mpirun -np 20 lmp -in in.input -var fname network_10wt.atom -var seed 1 -var Tpp 60 -var iters 600000
```

`network_10wt.atom` is the 10 wt% network, `Tpp` the oscillation period and `iters` the number of shear steps.
Other initial configurations (`-npeg`: number of stars):

## Data

- `rheology/data/spectra/`: one file per network, one row per period: T, ω, G′, G″, analysed cycles, p, largest-component fraction, number of stars.
- `rheology/data/experimental_G.txt`: experimental plateau modulus (wt%, kPa).
- `calibration/`: polymer-chain scaling (Fig. 4) and active-bond fraction vs concentration, simulation and experiment (Fig. 5).
- `topology/graph_<N>.txt`: network graphs for N stars (Figs. 9, 10a–d); `topology/distributions/`: graphs used for the bond distributions.
