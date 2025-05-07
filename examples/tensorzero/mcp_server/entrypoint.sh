#!/bin/sh

echo "Entrypoint: Waiting for MinIO to be healthy..."

# Simple loop to check MinIO health endpoint (within the Docker network)
# Adjust timeout as needed
TIMEOUT=60
START_TIME=$(date +%s)
while ! curl -sf http://minio:9000/minio/health/live > /dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$(($CURRENT_TIME - $START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Entrypoint: Timeout waiting for MinIO!"
        exit 1
    fi
    echo "Entrypoint: MinIO not ready, sleeping..."
    sleep 2
done
echo "Entrypoint: MinIO is healthy."

echo "Entrypoint: Configuring mc client and creating bucket 'tensorzero'..."

# Configure mc to talk to the MinIO server using the service name
# Use --insecure because we are using http
mc --insecure alias set local http://minio:9000 user password

# Create the bucket if it doesn't exist
# Use --insecure because we are using http
mc --insecure ls local/tensorzero > /dev/null 2>&1 || mc --insecure mb local/tensorzero

echo "Entrypoint: Bucket 'tensorzero' check/creation complete."

echo "Entrypoint: Executing the main container command: $@"

exec "$@"
