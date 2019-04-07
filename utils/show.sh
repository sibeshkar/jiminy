#!/bin/bash

pids=$(xdotool search --class "vnc")
i=0
for pid in $pids; do
    xdotool windowactivate $pid
done