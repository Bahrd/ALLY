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

from numpy import array, argsort
from numpy.random import randint, seed
from scipy.spatial.distance import cdist
from scipy.stats import mode
from itertools import product

from PIL import Image
from VoronoiUtilities import save_image, ITT

### Try @home:
##  x, S = ((0.25, 0.25), (2.5, 2.5)), {(0, 0): 0, (1, 1): 1, (2, 2): 2}; plain_vanilla_knn_lp(x, S, 1, 2) 
@ITT
def plain_vanilla_knn_lp(x, S, k, p): 
    X = tuple(S.keys())
    D = cdist(array(x), array(X), 'minkowski', p = p)
    classes = []
    for distances in D: 
        k_nearest_neighbors = argsort(distances)[:k]
        knn = [S[X[neighbor]] for neighbor in k_nearest_neighbors]
        pattern_class = int(mode(knn).mode[0])
        classes.append(pattern_class)
    return classes

@ITT
def knn_lp_Voronoi(x, S, k, p, w, colors):
    # Visualization tools
    image = Image.new("RGB", (w, w)); img = image.load()
    
    # Extract patterns from the learning set S...
    X = tuple(S.keys())
    # ... and compute distances to every pattern w.r.t. a selected Lp function
    D = cdist(array(x), array(X), 'minkowski', p = p)

    # k-NN classifiers assign $x$ to the class whose index is
    # a mode of its k nearest learning patterns' classes
    for (mn, distances) in enumerate(D): 
        ## Note here we don't care about the distances, we only need the class indices
        #  'k_nearest_neighbors' is a set such indices and...
        k_nearest_neighbors = argsort(distances)[:k]
        #  ... 'knn' is a subset with k nearest neighbor patterns
        knn = [S[X[neighbor]] for neighbor in k_nearest_neighbors]
        #  The mode determines the class index the pattern $x$ will be assigned to...
        #  (note the ties can affect the shape)
        pattern_class = int(mode(knn).mode[0])

        ## The line below 'paints a picture' pixel-by-pixel using colors
        #  associated with classes
        m, n = x[mn]; img[m, n] = colors[pattern_class]
    # Here we save it "ad perpetuam rei memoriam"
    save_image(f'./images/{k}-N' + 'N-L{}@{}', image, p, sd)

# Colors for illustrative purposes
c_red, c_green, c_blue, c_yellow, c_black, c_gray, c_whitish, c_white = ((0xff, 0, 0), (0, 0xff, 0), 
																		 (0, 0, 0xff), (0xff, 0xff, 0),
																		 (0, 0, 0), (0x80, 0x80, 0x80), 
																		 (0xdd, 0xdd, 0xdd), (0xff, 0xff, 0xff))
colors = [c_white, c_whitish, c_gray, c_black, c_red]

# Exemplary (hyper-)parameters
k, p, N = 0b111, 0b10, 0b1_000_000
Hanan = False

# Patterns' coordinates (i.e. feature vectors of ${X_n} \in S_N$)
sd, w = 0x1_000_000, 0x100; seed(sd)
nx, ny, S = randint(0o10, w - 0o10, N), randint(0o10, w - 0o10, N), {}

# For a Hanan's plantation: use 'product' rather than 'zip'
patterns = product(nx, ny) if Hanan else zip(nx, ny)

## Patterns assigned at random to classes 
#  (a number of classes should not exceed the size of the 'colors' list)
for x, y in patterns: S[(x, y)] = randint(len(colors))

# Set of patterns {x} to be classified
x = tuple(product(range(w), range(w)))

### Et voilà!
knn_lp_Voronoi(x, S, k, p, w, colors)

# ... and a (nutshell) summary and fanfares!
print('seed =', sd) # A tribute to CDMA (and H. Lamar & G. Antheil 1942's invention)