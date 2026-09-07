import numpy as np
from env.centerline import Centerline
# straight line along +x, 100m
c = Centerline(np.array([[0.,0.,0.],[100.,0.,0.]]), spacing=2.0)
n = len(c.points) - 1
la = c.lookahead(n, [0., 5., 20.])          # at the very end
assert abs(la[0][0] - c.points[-1][0]) < 1e-6, la
assert abs(la[1][0] - (c.points[-1][0] + 5.)) < 1e-6, f"5m past should extrapolate, got {la[1]}"
assert abs(la[2][0] - (c.points[-1][0] + 20.)) < 1e-6, f"20m past should extrapolate, got {la[2]}"
# mid-line unchanged (no extrapolation branch)
mid = c.lookahead(5, [0., 10.])
assert abs(mid[1][0] - (c.s[5] + 10.)) < 1e-6, mid
print("lookahead extrapolates past the end; mid-line unchanged  OK")
