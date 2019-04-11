#!/usr/bin/env bash
docker run -itd -e TURK_DB='' -p 5899:5899 -p 84:6080 --privileged --ipc host --cap-add SYS_ADMIN sibeshkar/jiminywob:latest demonstration -e wob.mini.TicTacToe-v0