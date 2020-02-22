import os
os.system('rm run_*.pkl')
os.system('rm -rf data')

for _ in range(100):
    os.system('python scripts/run_match.py generate-train-algo/ generate-train-algo-enemy/')
