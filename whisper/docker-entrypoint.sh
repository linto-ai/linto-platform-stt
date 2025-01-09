#!/bin/bash
set -a

echo "RUNNING STT"

# Check model
echo "Checking model format ..."
if [ -z "$MODEL" ]
then
    echo "Model type not specified, choosing Whisper medium model"
    export MODEL=medium
fi

# Launch parameters, environement variables and dependencies check
if [ -z "$SERVICE_MODE" ]
then
    echo "ERROR: Must specify an environment variable SERVICE_MODE in [ http | task | websocket ] (None was specified)"
    exit -1
else
    if [[ "$SERVICE_MODE" == "http" && "$ENABLE_STREAMING" != "true" ]]
    then
        echo "RUNNING STT HTTP SERVER"
        python3 http_server/ingress.py --debug
    elif [ "$SERVICE_MODE" == "task" ]
    then
        if [[ -z "$SERVICES_BROKER" ]]
        then 
            echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
            exit -1
        fi
        nvidia-smi 2> /dev/null > /dev/null
        if [ $? -eq 0 ];then
            echo "GPU detected"
            GPU=1
            OPT="--pool=solo"
        else
            echo "No GPU detected"
            GPU=0
            OPT=""
        fi
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up" || exit 1
        echo "RUNNING STT CELERY WORKER"
        celery --app=celery_app.celeryapp worker $OPT -Ofair --queues=${SERVICE_NAME} -c ${CONCURRENCY} -n ${SERVICE_NAME}_worker@%h
    elif [[ ("$SERVICE_MODE" == "http" && "$ENABLE_STREAMING" == "true") || "$SERVICE_MODE" == "websocket" ]]
    then
        echo "Running Websocket server on port ${STREAMING_PORT:=80}"
        python3 websocket/websocketserver.py
    else
        echo "ERROR: Must specify an environment variable SERVICE_MODE in [ http | task | websocket ] (got SERVICE_MODE=$SERVICE_MODE)"
        exit -1
    fi
fi

echo "Service stopped"