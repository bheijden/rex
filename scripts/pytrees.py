import jumpy as jp
import numpy as onp

t1 = [[1, 2], [3, 4], [5, 6]]
t2 = ((7, 8), (9, 10), (11, 12))
t3 = [[True, True], [False, False], [True, True]]

t4 = onp.select(True, t1, t2)
t5 = onp.where(t3, t1, t2)
t6 = jp.where(t3, t1, t2)