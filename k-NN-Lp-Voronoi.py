### An implementation of k-NN algorithm
## See e.g.:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
# https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
##  Given a set of learning patterns $S_N$:
#   1. Create a vector of lp distances to patterns' feature vectors ${X_n}$
#   2. For all patterns of interest, $x$, find:
#      * indices of k closest patterns 
#      * a mode of class indices and assign it to $x$

from numpy import arange, array, argpartition, partition, zeros
from scipy.spatial.distance import cdist
from scipy.stats import mode
from itertools import product
from random import randrange as RA, seed

from PIL import Image
from VoronoiUtilities import save_image, ITT

@ITT
def knn_lp_Voronoi(x, X, k, w, p, pattern_classes):
    # Visualization tools
    image = Image.new("RGB", (w, w)); img = image.load()
    
    # Compute distances to every pattern w.r.t. a selected Lp function
    D = cdist(array(x), array(X), 'minkowski', p = p)

    # k-NN classifiers assign $x$ to the class whose index is
    # a mode of the k nearest learning patterns' classes
    for (mn, distance) in enumerate(D): 
        # 'k_nearest_neighbors' is a set of k nearest neighbor pattern's indices
        k_nearest_neighbors = argpartition(distance, k)
        # 'knn' is a set of k nearest neighbor pattern's classes
        knn = [S[X[neighbor]] for neighbor in k_nearest_neighbors[:k]]
        # The mode determines the class the patterns are assigned to
        pattern_class = int(mode(knn).mode[0])

        ## A line below, pixel-by-pixel, 'paints a picture'
        m, n = x[mn]; img[m, n] = pattern_classes[pattern_class]

    save_image(f'./images/{k}-N' + 'N-L{}@{}', image, p, sd)

k, p, N, Hanan = 0b101, 0b10, 0b1000000, False
sd, w = 0x1000000, 0x100

# For illustrative purposes
c_red, c_green, c_blue, c_yellow, c_black, c_gray, c_whitish, c_white = ((0xff, 0, 0), (0, 0xff, 0), 
																		 (0, 0, 0xff), (0xff, 0xff, 0),
																		 (0, 0, 0), (0x80, 0x80, 0x80), 
																		 (0xdd, 0xdd, 0xdd), (0xff, 0xff, 0xff))
colors = [c_white, c_whitish, c_gray, c_black, c_red]

# Generate patterns' coordinates (i.e. feature vectors of ${X_n} \in S_N$)
seed(sd); 
nx, ny, S = [RA(10, w - 10) for _ in range(N)], [RA(10, w - 10) for _ in range(N)], {}

## For a Hanan's plantation: use 'product' rather than 'zip'
patterns = product(nx, ny) if Hanan else zip(nx, ny)
# Assign patterns at random (a number of classes should equal the size of the 'colors' list
for x, y in patterns: S[(x, y)] = RA(len(colors))
# Create a set of patterns {x} to be classified, then extract a set {X} of learning patterns
x, X = tuple(product(range(w), range(w))), tuple(S.keys())

knn_lp_Voronoi(x, X, k, w, p, colors)