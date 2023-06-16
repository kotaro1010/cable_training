#!/bin/bash

docker build --build-arg USER_ID=$UID -t cable-bridge:3.0 .