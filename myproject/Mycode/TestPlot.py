import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork

sim = cv.load(f'E:/大论文相关/covasim/myproject/results/test.sim')

# sim.plot(to_plot={
#     'total_infections': ['cum_infections'],
#     '每日新增': ['new_infections', 'new_diagnoses'],
#     '死亡': ['cum_deaths', 'cum_known_deaths'],
# })

sim.plot(to_plot={
    'total_counts': ['n_susceptible','cum_infectious','cum_infections','cum_recoveries','n_preinfectious']
})