from chat2.irisdata import *
for t,marker,c in zip(range(3),">ox","rgb"):
   plt.scatter(features[target==t,0],
               features[target==t,1],
               marker=marker,
               c=c)
plt.show()
