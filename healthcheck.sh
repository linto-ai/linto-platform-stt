#!/usr/bin/env bash

set -eax

if [ "$SERVICE_MODE" = "http" ]
then
    curl --fail http://localhost:80/healthcheck || exit 1
else
    # Check GPU utilization
    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | grep -v '^0$'; then
        # GPU is being utilized, assuming healthy
        exit 0
    else
        if [ "$SERVICE_MODE" = "websocket" ]
        then
            nc -z localhost ${STREAMING_PORT:=80} && exit 0 || exit 1
        else
            celery --app=celery_app.celeryapp inspect ping -d ${SERVICE_NAME}_worker@$HOSTNAME || exit 1
        fi
    fi
fi
