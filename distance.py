### Lp distance functions implementation
##  See: https://stackoverflow.com/questions/71091850/structural-pattern-matching-and-infinity/
from math import sqrt, inf
from dataclasses import dataclass

# Just a helper data class/structure
@dataclass 
class i: nf = inf

def lp_distance(x, y, p): 
    match p:
        case 0.0:  return int(x != 0.0) + int(y != 0.0)
        case 0.5:  return (sqrt(abs(x)) + sqrt(abs(y)))**2.0
        case 1.0:  return abs(x) + abs(y)
        case 2.0:  return sqrt(x**2.0 + y**2.0)
        case i.nf: return max(abs(x), abs(y))
        case    _: return pow(pow(abs(x), p) + pow(abs(y), p), 1.0/p)