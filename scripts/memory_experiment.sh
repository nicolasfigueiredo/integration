#!/bin/zsh
echo "Starting experiment...\n"
python3 memory_experiment.py 0
sleep 10
echo "Another 45 files... (1st round)"
python3 memory_experiment.py 1
sleep 10
echo "Another 45 files... (1st round)"
python3 memory_experiment.py 2
sleep 10
echo "Another 45 files... (1st round)"
python3 memory_experiment.py 3
sleep 10
echo "Another 45 files... (1st round)"
python3 memory_experiment.py 4