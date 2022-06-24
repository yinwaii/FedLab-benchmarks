#!/bin/bash

python3 server.py --ip 127.0.0.1 --port 3002 --world_size 3 --dataset mnist &

python3 client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 1 --dataset mnist &

python3 client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 2 --dataset mnist &

wait