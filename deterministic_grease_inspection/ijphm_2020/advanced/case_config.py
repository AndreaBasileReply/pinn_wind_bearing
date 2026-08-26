"""Case configuration for the 10 random-plane initializations of the IJPHM-2020 study.

The paper (section 4) initializes the MLP with 10 different randomly generated
planes ("case #1" .. "case #10").  The original run01-run04 scripts handle a
single, unnamed case.  Setting the environment variable ``CASE`` to an integer
makes every artifact of a run carry a ``_case<N>`` suffix so the 10 cases can
coexist; leaving ``CASE`` unset reproduces the original filenames exactly.

``CASE`` also seeds the random plane generator so the study is reproducible.
"""
import os

CASE = os.environ.get('CASE')

# Set JUDGMENT=1 to constrain the plane coefficients as described in section 3.4
# (engineering judgment) instead of the unconstrained np.random.random(4) draw.
JUDGMENT = os.environ.get('JUDGMENT', '0') == '1'

if CASE is None:
    SUFFIX, SEED = '', None
elif JUDGMENT:
    SUFFIX, SEED = '_ej%s' % CASE, 3000 + int(CASE)
else:
    SUFFIX, SEED = '_case%s' % CASE, 1000 + int(CASE)
# Set RUN30=0 to skip the (expensive) 30-year forecast at the end of run03.
RUN30 = os.environ.get('RUN30', '1') != '0'
