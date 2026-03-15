import re


def norm(s):
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub("\s+", " ", s)
    return s


def split(s):
    s = norm(s)
    if s == "":
        return []
    parts = s.split(" ")
    return parts


def join(xs):
    out = ""
    i = 0
    n = len(xs)
    while i < n:
        if i == 0:
            out = xs[i]
        else:
            out = out + " " + xs[i]
        i += 1
    return out
