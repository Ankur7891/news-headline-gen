
def overlap(a, b):
    ca = {}
    cb = {}
    i = 0
    while i < len(a):
        w = a[i]
        if w in ca:
            ca[w] = ca[w] + 1
        else:
            ca[w] = 1
        i += 1
    i = 0
    while i < len(b):
        w = b[i]
        if w in cb:
            cb[w] = cb[w] + 1
        else:
            cb[w] = 1
        i += 1
    inter = 0
    for k in ca:
        if k in cb:
            if ca[k] < cb[k]:
                inter = inter + ca[k]
            else:
                inter = inter + cb[k]
    pa = 0.0
    ra = 0.0
    fa = 0.0
    if len(a) > 0:
        pa = inter / float(len(a))
    if len(b) > 0:
        ra = inter / float(len(b))
    if pa + ra > 0:
        fa = 2.0 * pa * ra / (pa + ra)
    return pa, ra, fa
