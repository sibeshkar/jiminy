#!/bin/bash

echo "Launching windows"

#change hostname here

host=localhost

nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5900 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5901 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5902 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5903 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5904 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5905 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5906 &
nohup vncviewer -geometry 400x400 -ViewOnly -passwd ~/.vnc/passwd $host::5907 &


sleep 2

echo "Moving windows"


pids=$(xdotool search --class "vnc")
i=0
for pid in $pids; do
    #echo $pid
    #echo $(xdotool getwindowname $pid)
    #if [[ $(($lenPT % 2)) == 0 ]] then
    k=$(($i % 4))
    x_axis=$(($k * 500))
    
    if [[ $i -ge 4 ]]; then
        xdotool windowmove $pid $(($x_axis)) 700
        echo $(xdotool getwindowname $pid) "-->" $x_axis 700
    else
        xdotool windowmove $pid $(($x_axis)) 0
        echo $(xdotool getwindowname $pid) "-->" $x_axis 700
    fi
    # else
    #     x_axis = $i % 4 * 500
    #     xdotool windowmove $pid $x_axis 600
    # xdotool windowmove $pid $(($i)) 0
    # echo $i
    i=$(expr $i + 1)
done

