# [2021-09-02 clo] I've always found passing around data at different levels of hierarchy to be annoying.
# This lets us just go ahead and pass whatever we want around without worry.


# [2021-10-05] This controls whether we use zodb or not. Right now, I'm locking it behind a feature flag so that
# we can easily turn it on and off when we want to. For now I'm default it to off to not affect anything.
global use_zodb
use_zodb = False

global collector
collector = None

global blackboard
blackboard = {}

global param
param = None