from enum import Enum

class Country(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Country.RED.name)