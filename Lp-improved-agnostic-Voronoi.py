## The Lp agnostic 1-NN algorithm illustration [Project «⅄⅃LY»]
# 1. Generate N random patterns inside a square
# 2. Create the Voronoi's diagram for Lp, 0 < p ≤ 2 (p = 2.0 is somehow distingushed)
# 3. Compute an N × N Cartesian product of pattern's coordinates to get a Hanan grid
# 4. Associate the new N × (N - 1) patterns to the classes w.r.t. the selected Lp
# 5. Generate o Voronoi diagram for these N × N patterns (effectively, for the N × N Hanan grid)
# 6. Repeat the steps #2-#5 for other Lq, 0 < q ≤ 2 (q = 0.25 seems intriguing enough)
# 7. In order to improve approximation quality one can add a new (set of L) 
#    pattern(s) in the arbitrary position(s), together with the accompanying lattice points.

## &"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" - YPMV (Your Path May Vary)
#  .\Lp-improved-agnostic-Voronoi.py '2**3' '2**4 * 3 * 5 * 19 * 21673' - as used in the IEEE article 
#  Note 2**3 = 8 and 2**4 * 3 * 5 * 19 * 21673 = 98828880 - so ♫♪ please don't take [it serious] just because you can! ♪♫

from winsound import Beep as beep
from random import randrange as RA, seed
from sys import argv
from glob import glob; from os import remove
## Local stuff...
from VoronoiUtilities import *

## Voronoi diagrams and their agnostic versions...
rsd = int(RA(0x12345678))	
c, sd = (eval(argv[1]), eval(argv[2])) if len(argv) >= 3 else (eval(argv[1]), rsd) if len(argv) == 2 else (0x10, rsd)

# Lp, for p and q 
p, q, w = 2.0, 1.0, 0x100
## The reference diagrams for p and q...
NXY, _ = lp_Voronoi(w, p, c, sd), lp_Voronoi(w, q, c, sd)
NXY = list(NXY) # ♫♪ Ad futuram rei memoriam ♪♫
# ... together with the diagram's Lp-agnostic counterparts w.r.t. p and q
for pq in ((p, q), (q, p)):	lp_agnostic_Voronoi(*NXY, *pq, c, sd)

## ... and (supposedly) their more accurate versions
# Make a difference (to compute the improved version for Lp)
lp_Voronoi_set_op(p, sd, improved = False); lp_improved_agnostic_Voronoi(*NXY, 0x1, c, p, q, sd)
# Make another difference (to show the improvement)
lp_Voronoi_set_op(p, sd, improved = True)

## A bit (or two) of clean up...
for file in glob('.\images\*.png'): remove(file)
# ... and a (nutshell) summary and fanfares!
print('seed =', sd)
for pl in ((0x1b8, 0x7d), (0x1b8, 0x7d), (0x19f, 0x7d), (0x1b8, 0xfa)): beep(*pl) # ♫♪ ¡⅄⅃LY! ♪♫