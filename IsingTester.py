from IsingModel import IsingModel
import numpy as np
import random

lattice = IsingModel(size=(50, 50), temp=1.0, ini='r', dynamics="kawasaki")
lattice.run_animation(sweeps=10000, it_per_sweep=2500)
