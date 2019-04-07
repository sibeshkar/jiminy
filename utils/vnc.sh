#!/bin/bash

echo "Launching windows"

nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5900 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5901 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5902 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5903 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5904 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5905 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5906 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd localhost::5907 &


sleep 2

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

