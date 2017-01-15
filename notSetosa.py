from irisdata import *
is_setosa=(lables=='setosa')
features=features[~is_setosa]
labels=lables[~is_setosa]
virginica=(labels=='virginica')

best_acc=-1.0
best_fi=-1.0
best_t=-1.0

for fi in range(features.shape[1]):
    thresh=features[:,fi].copy()
    thresh.sort()

    for t in thresh:
        pred = (features[:,fi]>t)
        acc = (labels[pred]=='virginica').mean()
        if acc>best_acc:
            best_acc=acc
            best_fi=fi
            best_t=t
