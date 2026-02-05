from covasim import interventions
print([x for x in dir(interventions) if not x.startswith('_')])