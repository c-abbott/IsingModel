import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
import random


class IsingModel(object):
    def __init__(self, size, temp, ini, dynamics):
        """
            Ising Model class object.

            Attributes:
            size = size (tuple), dimensions of Ising lattice.
            temperature = temp (float), temperature of lattice.
            initial_state = ini (str), initial state of lattice
            dynamics = dynamics (str), Glauber or Kawasaki Dynamics
        """
        self.size = size
        self.temp = float(temp)
        self.ini = ini
        self.dynamics = dynamics
        self.build_lattice()

    def build_lattice(self):
        """
            Creates an ndarray of lattice sites, each site is
            occupied by either a spin up or spin down state.
        """
        # Random config.
        if self.ini == "r":
            self.lattice = np.random.choice(a=[-1, 1], size=self.size)
            self.total_E = self.get_total_E()
        # All up config.
        if self.ini == "u":
            self.lattice = np.ones(self.size, dtype=int)
            self.total_E = self.get_total_E()
        # Half up, half down config.
        if self.ini == "k":
            self.lattice = np.ones(self.size, dtype=int)
            a = -np.ones((50, 25), dtype=int)
            self.lattice[:, 25:] = a
            self.total_E = self.get_total_E()

    def pbc(self, indices):
        """
            Applies periodic boundary conditions (pbc) to a
            2D lattice.
        """
        return(indices[0] % self.size[0], indices[1] % self.size[1])

    def get_dE(self, indices):
        """
            Calculates the unit energy at a
            single lattice point on the 2D Ising lattice.
        """
        i, j = indices
        dE = 2 * self.lattice[i, j] * (
            self.lattice[self.pbc((i - 1, j))]
            + self.lattice[self.pbc((i + 1, j))]
            + self.lattice[self.pbc((i, j - 1))]
            + self.lattice[self.pbc((i, j + 1))]
        )
        return dE

    def unit_E(self, indices):
        """
            Calculating the energy value for a single
            point on the lattice (no flipping).
        """
        i, j = indices
        unit_E = - self.lattice[i, j] * (
            self.lattice[self.pbc((i - 1, j))]
            + self.lattice[self.pbc((i + 1, j))]
            + self.lattice[self.pbc((i, j - 1))]
            + self.lattice[self.pbc((i, j + 1))]
        )
        return unit_E

    def get_total_E(self):
        """
            Calculating the total energy of the
            whole Ising lattice.
        """
        total_E = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                total_E += self.unit_E((i, j))
        return (total_E / 2.0)

    def get_avg_obs(self, observables):
        """
            A function to calculate the average of a list
            of observables.
        """
        return np.mean(observables)

    def get_heat_capacity(self, energies):
        """
            A function to calculate the heat capacity given
            a list of energies.
        """
        heat_cap = (np.var(energies) / (self.size[0]*self.size[1]*self.temp**2))
        return heat_cap

    def get_abs_M(self):
        """
            Calculating total magnetisation due to all spins
            on the 2D Ising Lattice.
        """
        return (abs(np.sum(self.lattice)))

    def get_chi(self, mags):
        """
            Calculates susceptibility of Ising lattice given a list
            of magnetisations.
        """
        chi = (np.var(mags) / (self.size[0]*self.size[1]*self.temp))
        return chi

    def metropolis(self, dE):
        """
            Implementation of Metropolis algorithm to
            propagate Markov chain forwards in time.
        """
        if dE < 0:
            return True

        elif dE >= 0:
            prob = min(1, m.exp(- dE / self.temp))
            r = np.random.random()
            if r <= prob:
                return True
            elif r > prob:
                return False

    def glauber(self):
        """
            Glauber dynamics.
        """
        indices = (np.random.randint(0, self.size[0]),
                   np.random.randint(0, self.size[1]))

        dE = self.get_dE(indices)

        outcome = self.metropolis(dE)

        if outcome == True:
            self.lattice[indices] *= -1
            self.total_E += dE

    def kawasaki(self):
        """
            Kawasaki dynamics.
        """
        indices_1 = (np.random.randint(0, self.size[0]),
                     np.random.randint(0, self.size[1]))
        indices_2 = (np.random.randint(0, self.size[0]),
                     np.random.randint(0, self.size[1]))

        if self.lattice[indices_1] != self.lattice[indices_2]:
            dE_1 = self.get_dE(indices_1)
            self.lattice[indices_1] = -1 * self.lattice[indices_1]
            dE_2 = self.get_dE(indices_2)
            self.lattice[indices_1] = -1 * self.lattice[indices_1]
            dE = dE_1 + dE_2

            outcome = self.metropolis(dE)

            if outcome == True:
                self.lattice[indices_1] *= -1
                self.lattice[indices_2] *= -1
                self.total_E += dE

    def bootstrap(self, energies, samples):
        """
            Bootstrap method for generating error
            values assocaited with heat capacities.
        """
        error_data = []
        for i in range(samples):
            sampling_data = []
            for j in range(len(energies)):
                r = np.random.randint(0, (len(energies)-1))
                sampling_data.append(energies[r])
            error_data.append(self.get_heat_capacity(sampling_data))
        return m.sqrt(np.var(error_data))

    def bootstrap_chi(self, mags, samples):
        """
            Bootstrap method for generating error
            values assocaited with heat capacities.
        """
        error_data = []
        for i in range(samples):
            sampling_data = []
            for j in range(len(mags)):
                r = np.random.randint(0, (len(mags)-1))
                sampling_data.append(mags[r])
            error_data.append(self.get_chi(sampling_data))
        return m.sqrt(np.var(error_data))

    def animate(self, *args):
        """
            Creates, saves and returns image of the current state of
            Ising lattice for the FuncAnimation class.
        """
        for i in range(self.it_per_sweep):
            if self.dynamics == "glauber":
                self.glauber()
            elif self.dynamics == "kawasaki":
                self.kawasaki()
            elif self.dynamics == "kawasaki_2":
                self.kawasaki_2()
        self.image.set_array(self.lattice)
        return self.image,

    def run_animation(self, sweeps, it_per_sweep):
        """
            Used in partnership with the tester file
            to run the simulation.
        """
        self.it_per_sweep = it_per_sweep
        self.figure = plt.figure()
        self.image = plt.imshow(self.lattice, cmap='jet', animated=True)
        self.animation = animation.FuncAnimation(self.figure, self.animate, repeat=False, frames=sweeps, interval=50, blit=True)
        plt.colorbar(ticks=np.linspace(-1,1,2))
        plt.show()
