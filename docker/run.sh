#!/bin/bash

# MountDir=/home/beer/effect/cable_crack_detection
MountDir=/home/effect023/motoyama/cable_tyodai/training_testing
MountPosition=/home/appuser/cable_bridge

MountDir_video=/home/effect023/motoyama/cable_tyodai/videos
MountPosition_video=/home/appuser/cable_bridge/videos
Image=cable-bridge:3.0

# remove mode
# docker run -it --rm --gpus all -v $MountDir:$MountPosition -t $Image /bin/bash

# leaving mode
docker run -it --gpus all --name chodai_crack_detection_v3_torch \
    -v $MountDir:$MountPosition \
    -v $MountDir_video:$MountPosition_video \
    -t $Image /bin/bash

# daemon mode
# docker run -d --gpus all --name chodai_crack_detection_v3_torch -v $MountDir:$MountPosition -t $Image /bin/bash