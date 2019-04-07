#!/bin/bash

echo "Moving windows"


pids=$(xdotool search --class "vnc")
i=0
for pid in $pids; do

    k=$(($i % 4))
    x_axis=$(($k * 500))
    
    if [[ $i -ge 4 ]]; then
        xdotool windowmove $pid $(($x_axis)) 700
        echo $(xdotool getwindowname $pid) "-->" $x_axis 700
    else
        xdotool windowmove $pid $(($x_axis)) 0
        echo $(xdotool getwindowname $pid) "-->" $x_axis 700
    fi

    i=$(expr $i + 1)
done

