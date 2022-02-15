### Lp distance functions implementation
## See: https://stackoverflow.com/questions/71091850/structural-pattern-matching-and-infinity/
from math import sqrt, inf

#  Mimicking pattern matching
dist_dict = { 0.0: lambda x, y: int(x != 0.0) + int(y != 0.0),		# p == 0.0, the Hamming distance
			  0.5: lambda x, y: (sqrt(abs(x)) + sqrt(abs(y)))**2.0,	# p == 0.5, hand-crafted optimization
			  1.0: lambda x, y: abs(x) + abs(y),					# p == 1.0, the taxi-cab metric (Manhattan distance) 
		  	  2.0: lambda x, y: sqrt(x**2.0 + y**2.0),				# p == 2.0, the good ol' Euclid
		      inf: lambda x, y: max(abs(x), abs(y))}			    # p ==  ∞,  the max metric
def lp_distance(x, y, p): 
    try:			 return dist_dict[p](x, y)	# kinda branch-less programming ;)
    except KeyError: return pow(pow(abs(x), p) + pow(abs(y), p), 1.0/p)