## The Lp agnostic 1-NN algorithm illustration [Project «⅄⅃LY»]
# 1. Generate N random patterns inside a square
# 2. Create the Voronoi's diagram for Lp, 0 < p ≤ 2 (p = 2.0 is somehow distingushed)
# 3. Compute an N × N Cartesian product of pattern's coordinates to get a Hanan grid
# 4. Associate the new N × (N - 1) patterns to the classes w.r.t. the selected Lp
# 5. Generate o Voronoi diagram for these N × N patterns (effectively, for the N × N Hanan grid)
# 6. Repeat the steps #2-#5 for other Lq, 0 < q ≤ 2 (q = 0.25 seems unpredictable enough)

from time import time as TT
from random import randrange as RA, seed
from numpy import arange
from itertools import product, permutations
from math import sqrt, inf
from PIL import Image, ImageMath
from os.path import isfile

## A decorative fun... See: https://www.geeksforgeeks.org/decorators-in-python/
def ITT(f):
	def time_warper_wrapper(*args, **kwargs): 
		begin = TT() # from time import time as TT
		r = f(*args, **kwargs) 
		end = TT()
		print('{} created [in {}s]'.format(f.__name__, round(end - begin, 1)))
		return r
	return time_warper_wrapper

## Some random random utilities... (s = 0x96 for binary B&W diagrams and s = 0x20 for others)
random_xy = lambda l: [RA(int(l/0x20), int(0x1f * l/0x20)) for _ in range(2)]

def random_rbw(l = 0x32, u = 0xd0, s = 0x20): 
    rbw = [RA(l, u, s)] * 0x3 if RA(0x10) > 1 else c_red
    return rbw 
def random_rgb(l = 0x0, u = 0xff, s = 0x20): 
    rgb = [RA(l, u, s) for _ in range(0x3)]
    return rgb

# Insert 'False' to get ♫♪ True colors ♪♫...
random_color = random_rbw if(True) else random_rgb                  

## Colors re-united...
c_red, c_green, c_blue, c_yellow, c_black, c_gray, c_whitish, c_white = ((0xff, 0, 0), (0, 0xff, 0), 
																		 (0, 0, 0xff), (0xff, 0xff, 0),
																		 (0, 0, 0), (0x80, 0x80, 0x80), 
																		 (0xdd, 0xdd, 0xdd), (0xff, 0xff, 0xff))

## An lp-distance function implementation... See: https://www.geeksforgeeks.org/python-infinity/
#  (So ♫♪I long for♪♫ pattern matching... Python 3.10+?)
dictum_acerbum = {	0.0: lambda x, y: int(x != 0.0) + int(y != 0.0),		# p == 0.0, the Hamming distance
					0.5: lambda x, y: (sqrt(abs(x)) + sqrt(abs(y)))**2.0,	# p == 0.5, hand-crafted optimization
					1.0: lambda x, y: abs(x) + abs(y),						# p == 1.0, the taxi-cab metric (Manhattan distance) 
					2.0: lambda x, y: sqrt(x**2.0 + y**2.0),				# p == 2.0, the good ol' Euclid
					inf: lambda x, y: max(abs(x), abs(y))}					# p ==  ∞,  the max metric
def lp_distance(x, y, p): 
	try:				return dictum_acerbum[p](x, y)						# kinda branch-less programming ;)
	except KeyError:	return pow(pow(abs(x), p) + pow(abs(y), p), 1.0/p)	

## A naïve implementation of the 1-NN algorithm
#  (based on https://rosettacode.org/wiki/Voronoi#Python)
def classify_nn(w, p, img, nxy, colors):				
	if type(nxy) != (list or tuple): nxy = list(nxy) # Duct taped duck types (a.k.a. ♫♪ Disposable Heroes ♪♫)
	for x, y in product(range(w), range(w)):
		dmin, j = lp_distance(w, w, p), 0
		for i, (nx, ny) in enumerate(nxy):
			d = lp_distance(nx - x, ny - y, p)
			if d < dmin: 
				dmin, j = d, i
		img[x, y] = tuple(colors[j])

## Pinpointing patterns
def pin_patterns(img, nxy, motif, color):
	for nx, ny in nxy:
		for dx, dy in product(motif, motif): # (Let's Cartesian'em all!)
			img[nx + dx, ny + dy] = color

## Save'm all...
# (D'u know, subroutines were invented (for a reason ;) about 70 years ago? 
# (D.J. Wheeler 1952: https://youtu.be/ImLFlLjSveM?t=404)
def save_image(path, im_file, p, sd):
	f = path.format(p, sd)
	im_file.save(f + '.png', 'PNG'); im_file.save(f + '.pdf', 'PDF')

### Selection of 2D Voronoi diagrams generators
##  A diagram of seeds (patterns) planted on a Hanan grid
def lp_planted_Voronoi(sd, w = 0x100, p = 2.0, Hanan = False, context = True):
	seed(sd) # Controlled randomness to get a better picture of the phenomenon
	         # ♫♪ Choking on the bad, bad, bad, bad, bad, bad seed! ♪♫
	
	## Pattern classes' colors
	colors = [c_red, c_whitish, c_gray, c_black, c_black, c_whitish]

	# ♫♪ We're gonna have to reap from some seed that's been sowed... ♪♫
	pp = [RA(0x10, w - 0x10), RA(0x10, w - 0x10)]        # Our 'pater noster' mother-pattern...
	planted = list(permutations(pp))				     # ... and the original kid-like patterns...
	implanted = list(product(pp, pp))				     # ... and the neighborhood ones
	if Hanan == False:
		outgrid = [pp[1], RA(w)]					     # A stray pattern ♫♪ Just like the curse, 
													     # just like the stray, you feed it once and now it stays! ♪♫ 
		planted += [outgrid]; implanted += [outgrid]
		if context == True:	implanted += [[pp[0], outgrid[1]]] # An on-grid companion of the stray one 
														       # (♫♪ 'cause misery loves company! ♪♫)
	## Filling cells (i.e. performing classification)
	image = Image.new("RGB", (w, w)); img = image.load()
	classify_nn(w, p, img, implanted, colors)
	save_image('./images/Voronoi-planted-L{}@{}', image, p, sd)
	## ... and painting patterns
	pin_patterns(img, implanted,      [-1, 0, 1], c_white)
	pin_patterns(img, planted, [-2, -1, 0, 1, 2], c_yellow)
	save_image('./images/Voronoi-planted-sites-L{}@{}', image, p, sd)

### A (hard) working stuff... 
## 1. lp_Voronoi				    - set $S_{N}$ 
## 2. lp_agnostic_Voronoi		    - set $A_{N}$
## 3. lp_improved_agnostic_Voronoi  - set $A_{N+L}$ 
###   lp_Voronoi_set_op             - with a gim[p]hotoshop gimmick... e.g. to perform a visual $S \setminus A$

@ITT
def lp_Voronoi(w = 0x100, p = 2.0, c = 0x10, sd = 0x303):
	seed(sd) # Controlled randomness that yields the same pseudo-random patterns for various p
			 # Just a standard random case... # Black (, red) & white(-ish)...
	
	## Creating patterns
	nxy, nrgb = zip(*((random_xy(w), random_color(0x0, 0x100)) for _ in range(c)))	

	## Drawing cells... (i.e. classifying w.r.t. the set Sn)
	image = Image.new("RGB", (w, w)); img = image.load()
	classify_nn(w, p, img, nxy, nrgb)
	save_image('./images/Voronoi-L{}@{}', image, p, sd)

	## ... and patterns
	pin_patterns(img, nxy, [-2, -1, 0, 1, 2], c_yellow)
	save_image('./images/Voronoi-sites-L{}@{}', image, p, sd)

	return zip(*nxy)

@ITT
def lp_agnostic_Voronoi(NX, NY, p = 2.0, q = 0.25, c = 0x10, sd = 0x303):
	## Essentially, given N points, we 'yield' a Cartesian product of two vectors 
	# of N elements being their first and second coordinates, respectively
	f = './images/Voronoi-L{}@{}'.format(p, sd)
	image = Image.open(f + '.png')
	img, (w, _) = image.load(), image.size
	nrgb = [img[nx, ny] for nx, ny in product(NX, NY)]
	image.close()

	## Drawing cells... (i.e. classifying w.r.t. the set An)
	image = Image.new("RGB", (w, w)); img = image.load()
	classify_nn(w, p, img, product(NX, NY), nrgb)
	save_image('./images/Lp-agnostic-Voronoi-L{}@{}', image, p, sd)

	## ... and patterns
	pin_patterns(img, product(NX, NY),    [-1, 0, 1], c_white)
	pin_patterns(img, zip(NX, NY), [-2, -1, 0, 1, 2], c_yellow)
	save_image('./images/Lp-agnostic-Voronoi-sites-L{}@{}', image, p, sd)

@ITT
def lp_improved_agnostic_Voronoi(NX, NY, m = 0x1, c = 0x10, p = 2.0, q = 0.25, sd = 0x303):
	## Generate extra patterns for extra precision (in locations
	#   where the classifiers differ for Lp and for agnostic-Lp).
	## ⅄⅃LY 
	f = './images/agnostic-Voronoi-math-L{}@{}'.format(p, sd)
	image = Image.open(f + '.png') 
	img, (w, _)  = image.load(), image.size
	ax, ay = [], []
	while(len(ax) < m * c):
		x, y = random_xy(w)
		if(img[x, y] != (0x0, 0x0, 0x0)): ax += [x]; ay += [y]
	image.close()
	NX += tuple(ax); NY += tuple(ay)

	# Set pattern's classes from the image
	f = './images/Voronoi-L{}@{}'.format(p, sd)
	image = Image.open(f + '.png'); img = image.load()
	nrgb = [img[nx, ny] for nx, ny in product(NX, NY)]
	image.close()
	
	## Filling... (i.e. classifying w.r.t. the set An+l)
	image = Image.new("RGB", (w, w)); img = image.load()
	classify_nn(w, p, img, product(NX, NY), nrgb)
	save_image('./images/Lp-improved-agnostic-Voronoi-L{}@{}', image, p, sd)
	## ... and painting patterns	
	pin_patterns(img, product(NX, NY),           [-1, 0, 1], c_white)
	pin_patterns(img, zip(NX, NY),     	  [-2, -1, 0, 1, 2], c_yellow)
	pin_patterns(img, zip(ax, ay), [-3, -2, -1, 0, 1, 2, 3], c_red)
	save_image('./images/Lp-improved-agnostic-Voronoi-sites-L{}@{}', image, p, sd)

## Perform an 'operation' on diagrams (a set difference $S \setminus A$ in particular, when "operation = 'abs(a - b)'"
def lp_Voronoi_set_op(p = 2, sd = 0x303, improved = False, operation = 'abs(a - b)'):
	fa = './images/Lp-{}agnostic-Voronoi-L{}@{}.png'.format('improved-' if improved else '', p, sd)
	if(isfile(fa)):
		fp = './images/Voronoi-L{}@{}.png'.format(p, sd)
		fav, fpv = Image.open(fa), Image.open(fp); fdv = Image.new('RGB', fpv.size)
		#ImageMath doesn't process RGB images (at the time of writing this code)...
		fdv = Image.merge('RGB', [ImageMath.eval('convert({}, "L")'.format(operation), a = ipb, b = iqb) \
								  for (ipb, iqb) in zip(fpv.split(), fav.split())])
		save_image('./images/Agnostic-{}'.format('improved-' if improved else '') + 'Voronoi-math-L{}@{}', fdv, p, sd)
	else:
		print('File: [{}] not found!'.format(fa))