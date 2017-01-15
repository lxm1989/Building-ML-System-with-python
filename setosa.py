from irisdata import *

plength=features[:,2]
is_setosa=(lables=='setosa')

max=plength[is_setosa].max()
min=plength[~is_setosa].min()
print('Max of setosa:{0}.'.format(max))
print('Min of others:{0}.'.format(min))



