#!/bin/bash
set -ea

echo "RUNNING STT"

# Set default UID and GID (defaults to www-data: 33:33 if not specified)
USER_ID=${USER_ID:-33}
GROUP_ID=${GROUP_ID:-33}

# Default values for user and group names
USER_NAME="appuser"
GROUP_NAME="appgroup"

# Function to create a user/group if needed and adjust permissions
function setup_user() {
    echo "Configuring runtime user with UID=$USER_ID and GID=$GROUP_ID"

    # Check if a group with the specified GID already exists
    if getent group "$GROUP_ID" >/dev/null 2>&1; then
        GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
        echo "A group with GID=$GROUP_ID already exists: $GROUP_NAME"
    else
        # Create the group if it does not exist
        echo "Creating group with GID=$GROUP_ID"
        groupadd -g "$GROUP_ID" "$GROUP_NAME"
    fi

    # Check if a user with the specified UID already exists
    if id -u "$USER_ID" >/dev/null 2>&1; then
        USER_NAME=$(getent passwd "$USER_ID" | cut -d: -f1)
        echo "A user with UID=$USER_ID already exists: $USER_NAME"
    else
        # Create the user if it does not exist
        echo "Creating user with UID=$USER_ID and GID=$GROUP_ID"
        useradd -m -u "$USER_ID" -g "$GROUP_NAME" "$USER_NAME"
    fi

    # Adjust ownership of the application directories
    echo "Adjusting ownership of application directories"
    chown -R "$USER_NAME:$GROUP_NAME" /usr/src/app
}

# Check model
echo "Checking model format ..."
if [ -z "$MODEL_TYPE" ]
then
    echo "Model type not specified, expecting Vosk Model"
    export MODEL_TYPE=vosk
fi

if  [ "$MODEL_TYPE" = "vosk" ]
then
    echo "Using Vosk format's model"

elif [ "$MODEL_TYPE" = "lin" ]
then
    echo "Processing model ... "
    ./lin_to_vosk.py
else
    echo "Unknown model type $MODEL_TYPE. Assuming vosk model"
fi

# Setup the runtime user
setup_user

# Launch parameters, environement variables and dependencies check
if [ -z "$SERVICE_MODE" ]
then
    echo "ERROR: Must specify an environment variable SERVICE_MODE in [ http | task | websocket ] (None was specified)"
    exit -1
else
    if [ "$SERVICE_MODE" = "http" ] 
    then
        echo "RUNNING STT HTTP SERVER"
        gosu "$USER_NAME" python http_server/ingress.py --debug
    elif [ "$SERVICE_MODE" == "task" ]
    then
        if [[ -z "$SERVICES_BROKER" ]]
        then 
            echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
            exit -1
        fi
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up"
        echo "RUNNING STT CELERY WORKER"
        gosu "$USER_NAME" celery --app=celery_app.celeryapp worker -Ofair --queues=${SERVICE_NAME} -c ${CONCURRENCY} -n ${SERVICE_NAME}_worker@%h

    elif [ "$SERVICE_MODE" == "websocket" ]
    then
        echo "Running Websocket server on port ${STREAMING_PORT:=80}"
        gosu "$USER_NAME" python websocket/websocketserver.py
    else
        echo "ERROR: Must specify an environment variable SERVICE_MODE in [ http | task | websocket ] (got SERVICE_MODE=$SERVICE_MODE)"
        exit -1
    fi
fi

echo "Service stopped"
