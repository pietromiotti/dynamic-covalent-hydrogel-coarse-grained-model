import numpy as np
import math
import click
from rotation import rotation
import random

random.seed(0)

class Network(object):
    box = None
    natomTypes = 2
    pos = None
    bonds = []
    nangleTypes = 1
    angles = None
    ndihedralTypes = 0
    dihedrals = None
    solventAtomType = 1

    def Network(self, nPEG, nMon, box_x, box_y, box_z, bondLength, angle, dihedral):
        self.box_x = box_x
        self.box_y = box_y
        self.box_z = box_z
        self.nPEG_1 = nPEG//2
        self.nPEG_2 = nPEG - self.nPEG_1
        self.nMon = nMon
        self.bondLength = bondLength
        self.angle = angle
        self.dihedral = dihedral
        self.natom = nPEG * (4*nMon + 1)
        self.nangle = nPEG * (4*nMon + 1 - 2)
        self.ndihedral = nPEG * (4*nMon - 3)
        self.natomTypes = 4
        self.nangleTypes = 1
        self.ndihedralTypes = 0
        self.solventAtomType = 1
        self.pos = np.zeros((self.natom, 3))
        self.types = np.zeros((self.natom, 1))
        self.mol = np.zeros((self.natom, 1))

        self.bonds = []
        self.angles = []
        self.positions = []
        self.dihedrals = np.zeros((self.ndihedral, 4), dtype=int)
        self.currentIdx = 0
        self.currentMol = 1

        self.NORMAL_ATOM = 1
        self.FUNCTIONAL_ATOM_1 = 2
        self.FUNCTIONAL_ATOM_2 = 3

        self.molecule_radius = self.nMon * self.bondLength

        self.generateLattice()
        self.generateNetwork()

        self.bonds = np.array(self.bonds)
        self.bonds += 1

        self.angles = np.array(self.angles)
        self.angles += 1

        self.types += 1

    def generateNetwork(self):
        self.generatePEG(self.FUNCTIONAL_ATOM_1, self.nPEG_1)
        self.generatePEG(self.FUNCTIONAL_ATOM_2, self.nPEG_2)
        self.angle = len(self.angles)

    def generateArm(self, central_idx, direction, sign, functionalType):
        queue = [central_idx]
        for i in range(self.nMon):
            if i == 0:
                self.pos[self.currentIdx, :] = self.pos[central_idx, :]
                self.bonds.append([central_idx, self.currentIdx])
            else:
                self.pos[self.currentIdx, :] = self.pos[self.currentIdx - 1, :]
                self.bonds.append([self.currentIdx - 1, self.currentIdx])

            self.pos[self.currentIdx, direction] += sign * self.bondLength

            if (i == self.nMon-1):
                self.types[self.currentIdx] = functionalType
            else:
                self.types[self.currentIdx] = self.NORMAL_ATOM
            self.mol[self.currentIdx] = self.currentMol
            queue.append(self.currentIdx)
            self.currentIdx += 1

    def generateSinglePEG(self, idx, functionalType):
        if idx < len(self.positions):
            center_pos = self.positions[idx].copy()
        else:
            center_pos = np.array([
                self.molecule_radius + np.random.rand() * (self.box_x - 2*self.molecule_radius),
                self.molecule_radius + np.random.rand() * (self.box_y - 2*self.molecule_radius),
                self.molecule_radius + np.random.rand() * (self.box_z - 2*self.molecule_radius)
            ])

        self.pos[self.currentIdx, :] = center_pos
        central_idx = self.currentIdx
        self.types[self.currentIdx] = self.NORMAL_ATOM
        self.mol[self.currentIdx] = self.currentMol

        self.currentIdx += 1

        first_arm = self.currentIdx
        self.generateArm(central_idx, 0, 1, functionalType)
        second_arm = self.currentIdx
        self.generateArm(central_idx, 0, -1, functionalType)
        third_arm = self.currentIdx
        self.generateArm(central_idx, 1, 1, functionalType)
        fourth_arm = self.currentIdx
        self.generateArm(central_idx, 1, -1, functionalType)

        randomNoise = np.random.rand(3) * 2 * np.pi

        molecule_positions = self.pos[central_idx:self.currentIdx, :].copy()

        for i in range(self.currentIdx - central_idx):
            atom_idx = central_idx + i
            if i == 0:
                continue

            relative_pos = molecule_positions[i, :] - center_pos

            rotated_pos = rotation(
                relative_pos,
                randomNoise[0],
                randomNoise[1],
                randomNoise[2]
            )

            new_pos = center_pos + rotated_pos

            new_pos[0] = np.mod(new_pos[0], self.box_x)
            new_pos[1] = np.mod(new_pos[1], self.box_y)
            new_pos[2] = np.mod(new_pos[2], self.box_z)

            self.pos[atom_idx, :] = new_pos

        self.currentMol += 1
        print(f"Generated PEG molecule {self.currentMol-1} at center {center_pos}")

    def generatePEG(self, functionalType, nPEG):
        for i in range(nPEG):
            self.generateSinglePEG(i, functionalType)

    def generateLattice(self):
        molecule_diameter = 2 * self.molecule_radius
        safety_margin = 0.5 * self.bondLength
        min_spacing = molecule_diameter + safety_margin

        print(f"INFO: Molecule radius: {self.molecule_radius:.2f}")
        print(f"INFO: Molecule diameter: {molecule_diameter:.2f}")
        print(f"INFO: Minimum spacing required: {min_spacing:.2f}")

        if min_spacing > min(self.box_x, self.box_y, self.box_z):
            print(f"WARNING: Box may be too small for molecules of this size!")
            print(f"INFO: Box dimensions: {self.box_x} x {self.box_y} x {self.box_z}")

        effective_box_x = self.box_x - 2 * self.molecule_radius
        effective_box_y = self.box_y - 2 * self.molecule_radius
        effective_box_z = self.box_z - 2 * self.molecule_radius

        nx = max(1, int(effective_box_x / min_spacing))
        ny = max(1, int(effective_box_y / min_spacing))
        nz = max(1, int(effective_box_z / min_spacing))

        print(f"INFO: Effective box size (after boundary margins): {effective_box_x:.2f} x {effective_box_y:.2f} x {effective_box_z:.2f}")
        print(f"INFO: Grid will be: {nx} x {ny} x {nz} = {nx*ny*nz} positions")

        if nx == 1:
            x = [self.box_x / 2]
        else:
            x_start = self.molecule_radius + effective_box_x / (2 * nx)
            x_end = self.box_x - self.molecule_radius - effective_box_x / (2 * nx)
            x = np.linspace(x_start, x_end, nx)

        if ny == 1:
            y = [self.box_y / 2]
        else:
            y_start = self.molecule_radius + effective_box_y / (2 * ny)
            y_end = self.box_y - self.molecule_radius - effective_box_y / (2 * ny)
            y = np.linspace(y_start, y_end, ny)

        if nz == 1:
            z = [self.box_z / 2]
        else:
            z_start = self.molecule_radius + effective_box_z / (2 * nz)
            z_end = self.box_z - self.molecule_radius - effective_box_z / (2 * nz)
            z = np.linspace(z_start, z_end, nz)

        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        self.positions = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

        np.random.shuffle(self.positions)

        print(f"INFO: Generated {len(self.positions)} lattice positions")

        for i, pos in enumerate(self.positions[:5]):
            print(f"INFO: Position {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            min_dist_to_wall = min(
                pos[0], self.box_x - pos[0],
                pos[1], self.box_y - pos[1],
                pos[2], self.box_z - pos[2]
            )
            print(f"INFO: Minimum distance to wall: {min_dist_to_wall:.2f}")

    def writeNetwork(self, output):
        with open(output, 'w') as f:

            file_comment = "file generated by GenerateNetwork.py"
            separator = '\n\n'
            f.write(file_comment + '\n\n')

            f.write(str(self.natom) + " atoms\n")
            f.write(str(len(self.bonds)) + " bonds\n")
            f.write(separator)

            f.write(str(4) + " atom types\n")
            f.write(str(2) + " bond types\n")

            f.write(separator)

            boxSize = "0.0 " + str(self.box_x) + " xlo xhi\n"
            boxSize += "0.0 " + str(self.box_y) + " ylo yhi\n"
            boxSize += "0.0 " + str(self.box_z) + " zlo zhi\n"
            boxSize += "0.0 0.0 0.0 xy xz yz\n"
            f.write(boxSize)
            f.write(separator)

            f.write("Masses")
            f.write(separator)
            f.write("1 1\n")
            f.write("2 1\n")
            f.write("3 1\n")
            f.write("4 1\n")
            f.write(separator)

            f.write("Atoms")
            f.write(separator)

            self.write_element(f, 'coord')
            f.write(separator)
            f.write("Bonds")
            f.write(separator)
            self.write_element(f, 'bonds')

    def write_element(self, fileP, mode):

        if mode == 'coord':
            iters = self.natom
        elif mode == 'bonds':
            iters = len(self.bonds)
        elif mode == 'angles':
            iters = len(self.angles)

        for index in range(0, iters):
            real_index = index + 1
            if mode == 'coord':
                fileP.write(str(real_index) + " " + str(int(self.mol[index][0])) + " " + str(int(self.types[index][0])) +
                            " " + str(self.pos[index, 0]) +
                            " " + str(self.pos[index, 1]) +
                            " " + str(self.pos[index, 2]) +
                            "\n"
                            )
            elif mode == 'bonds':
                fileP.write(str(real_index) + " " + str(1) +
                            " " + str(self.bonds[index, 0]) +
                            " " + str(self.bonds[index, 1]) +
                            "\n"
                            )

            elif mode == 'angles':
                fileP.write(str(real_index) + " " + str(1) +
                            " " + str(self.angles[index][0]) +
                            " " + str(self.angles[index][1]) +
                            " " + str(self.angles[index][2]) +
                            "\n"
                            )

    def visualizeNetwork(self, file):
        backup_mol = self.mol.copy().astype(int)
        backup_type = self.types.copy().astype(int)

        self.mol = self.types.copy().astype(int)
        self.types = backup_mol.copy().astype(int)

        self.writeNetwork(file)

        self.mol = backup_mol.copy().astype(int)
        self.types = backup_type.copy().astype(int)
