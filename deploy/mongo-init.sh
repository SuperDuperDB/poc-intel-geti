#!/bin/bash

set -eu

# replicate set initiate
echo "Checking mongo container..."
until mongosh --host mongodb --eval "print(\"waited for connection\")"; do
	sleep 1
done

echo "Initializing replicaset..."
mongosh --host mongodb <<EOF
    rs.initiate(
      {
          _id: "rs0",
          version: 1,
          members: [
            { _id: 0, host: "mongodb:27017"}
          ]
      }
    )
    rs.status()
EOF
