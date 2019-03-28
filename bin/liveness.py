#!/usr/bin/env python
# liveness.py - ping to check if RPC is alive

import time
try:
    # We write out a timestamp to a tmp file every time we get a diagnostics ping
    # This is a good sign the worker is healthy, so we'll use it as liveness check
    f = open('/tmp/jiminy-liveness')
except IOError:
    # gdb: If the file doesn't exist yet, for now just assume it's fine
    print('liveness file does not exist yet, meaning no client has connected to this environment')
    exit(1)

# Implicitly, this will throw an exception if the file doesn't exist,
# which will return a non-zero exit code, which indicates unhealthy-ness.
t = float(f.read().strip())

# Compare current time to this written out time
if time.time() - t > 360:
    # Last liveness write-out was more than 6 minutes ago
    # This should nominally happen every 5 minutes
    # see jiminy.envs.diagnostics.Network for the actual parameter
    # exit code indicates unhealthy worker
    print('liveness is too old - dead worker?')
    exit(1)

# For now that's all we'll check
print('liveness is good, check in again later')
exit(0)
