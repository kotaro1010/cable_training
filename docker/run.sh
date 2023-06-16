#!/bin/bash

MountDir=/home/effect025/hara/cable_bridge
MountPosition=/home/appuser/cable_bridge
Image=cable-bridge:3.0

# remove mode
# docker run -it --rm --gpus all -v $MountDir:$MountPosition -t $Image /bin/bash

# leaving mode
docker run -it --gpus all --name chodai_crack_detection_v3_torch  $NETWORK $PORT -v $MountDir:$MountPosition -t $Image /bin/bash