"""
Download NSFG (National Survey of Family Growth) raw data files.
We take only the data from Downey's repo — no helper classes.
Everything else we build from scratch.

Run once before starting Chapter 1:
    python data/download_nsfg.py
"""

import urllib.request
import os

BASE_URL = "https://raw.githubusercontent.com/AllenDowney/ThinkStats2/master/code/"

# Raw data only — no thinkstats2.py, no nsfg.py
FILES = [
    "2002FemPreg.dct",
    "2002FemPreg.dat.gz",
    "2002FemResp.dct",
    "2002FemResp.dat.gz",
]

data_dir = os.path.dirname(os.path.abspath(__file__))

print("Downloading NSFG data files...")
for filename in FILES:
    dest = os.path.join(data_dir, filename)
    if os.path.exists(dest):
        print(f"  already exists: {filename}")
        continue
    url = BASE_URL + filename
    print(f"  downloading: {filename}")
    urllib.request.urlretrieve(url, dest)

print("\nDone. All NSFG files saved to data/")
print("Start with: python ch01_eda.py")
