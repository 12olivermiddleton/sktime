from pyximport import install ; install()

from utils import Particle

a = Particle(1,2,3)
print(a.get_momentum())