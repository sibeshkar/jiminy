#!/bin/bash

pids=$(xdotool search --class "vnc")
i=0
for pid in $pids; do
    xdotool windowkill $pid
done