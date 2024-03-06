export SUPERDUPERDB_CLUSTER_CDC_URI='http://cdc:8001'
export SUPERDUPERDB_DATA_BACKEND="mongodb://mongodb:27017/geti"
export SUPERDUPERDB_CLUSTER_VECTOR_SEARCH='lance://vector-search:8000'

python -m superduperdb config
