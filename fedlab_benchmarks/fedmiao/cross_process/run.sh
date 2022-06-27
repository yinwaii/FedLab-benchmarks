#!/bin/bash

python3 server.py --ip 127.0.0.1 --port 3002 --world_size 4 --round 3 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 1 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 2 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 3 --dataset mnist &

wait