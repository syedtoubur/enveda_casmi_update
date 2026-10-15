import sys
sys.path.insert(0, 'E:/casmi/lines/exploit/build')
import test_patch_harness_dump as t
import casmi_post as cp
k = t.k
smis = ['C' * i + 'O' for i in range(1, 31)]
s = t.state()
s['base']['m3'] = (smis, [k(x) for x in smis], 0.95, list(range(30, 0, -1)), ['F'] * 30)
keyed = {k(smis[0]), k(smis[2])}
b = cp.run(s, t.deps(gl_reverse=False), **t.E006_ON)['m3']
o = cp.run(s, t.deps(gl_reverse=False), E180_DROP=True, E180_KEYS=keyed, **t.E006_ON)['m3']
print(len(b), len(o), keyed <= {k(x) for x in b}, bool(keyed & {k(x) for x in o}))
