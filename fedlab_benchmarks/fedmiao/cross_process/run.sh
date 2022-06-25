#!/bin/bash

python3 server.py --ip 127.0.0.1 --port 3002 --world_size 8 --sample 0.5 --dataset mnist &

python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 1 --dataset mnist &

python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 2 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 3 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 4 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 5 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 6 --dataset mnist &
python3 client.py --ip 127.0.0.1 --port 3002 --world_size 8 --rank 7 --dataset mnist &

wait