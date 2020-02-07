import numpy as np
import sys
from IsingModel import IsingModel
from matplotlib import pyplot as plt


def main():
    if len(sys.argv) != 4:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <parameters file>" + "<glauber output file>"
              + "kawasaki output file")
        quit()
    else:
        infile_parameters = sys.argv[1]
        glauber_file = sys.argv[2]
        kawasaki_file = sys.argv[3]

    glauber_output = open(glauber_file, "w+")
    kawasaki_output = open(kawasaki_file, "w+")

    # open input file and assinging parameters
    with open(infile_parameters, "r") as input_file:
        # read the lines of the input data file
        line = input_file.readline()
        items = line.split(", ")

        min_temp = float(items[0])  # Minimum temperature
        max_temp = float(items[1])  # Maximum temperature.
        temp_step = float(items[2]) # Temperature step.
        sweeps = int(items[3])      # No. of sweeps.
        eqm_sweeps = int(items[4])  # No. of EQM sweeps.
        n = int(items[5])           # How often to collect data
        samples = int(items[6])     # Bootstrap samples
        dynamics = str(items[7])    # Lattice dynamics.
        lattice_size = (int(items[8]), int(items[8]))  # Lattice size.

    # Domain for simulation.
    temp_range = np.arange(min_temp, max_temp + temp_step, temp_step)

    # List creation for data storage.
    energies = []
    magnetisations = []
    heat_capacties = []
    heat_cap_errors = []
    chis = []
    chi_errors = []

    # Simulation begins.
    for i in range(temp_range.size):
        lattice = IsingModel(size=lattice_size, temp=temp_range[i],
                            ini='u', dynamics=dynamics)
        energy_per_temp = []
        mag_per_temp = []

        # Glauber sweeps of lattice.
        if lattice.dynamics == "glauber":
            for j in range(sweeps):
                print(j)
                for k in range(lattice_size[0]*lattice_size[1]):
                    lattice.glauber()
                    if (j+1) >= eqm_sweeps and (j+1) % n == 0:
                        energy_per_temp.append(lattice.total_E)
                        mag_per_temp.append(lattice.get_abs_M())

            # Appending average values to list for file writer.
            energies.append(lattice.get_avg_obs(energy_per_temp))
            magnetisations.append(lattice.get_avg_obs(mag_per_temp))
            heat_capacties.append(lattice.get_heat_capacity(energy_per_temp))
            heat_cap_errors.append(lattice.bootstrap(energy_per_temp, samples))
            chis.append(lattice.get_chi(mag_per_temp))
            chi_errors.append(lattice.bootstrap_chi(mag_per_temp, samples))

            # Writing values to file.
            glauber_output.write('%lf, %lf, %lf, %lf, %lf, %lf, %lf\n' % (temp_range[i], energies[i], magnetisations[i],
                                                                          heat_capacties[i], heat_cap_errors[i], chis[i], chi_errors[i]))

        # Kawasaki sweeps of lattice.
        elif lattice.dynamics == "kawasaki":
            for l in range(sweeps):
                for m in range(lattice_size[0]*lattice_size[1]):
                    lattice.kawasaki()
                    if (l+1) >= eqm_sweeps and (l+1) % n == 0:
                        energy_per_temp.append(lattice.total_E)

            # Appending average values to list for file writer.
            energies.append(lattice.get_avg_obs(energy_per_temp))
            heat_capacties.append(lattice.get_heat_capacity(energy_per_temp))
            heat_cap_errors.append(lattice.bootstrap(energy_per_temp, samples))

            # Writing values to file.
            kawasaki_output.write('%lf, %lf, %lf, %lf\n' %
                                  (temp_range[i], energies[i], heat_capacties[i], heat_cap_errors[i]))

    # Closing files.
    glauber_output.close()
    kawasaki_output.close()


main()
