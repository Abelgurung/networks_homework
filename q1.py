# to run:

# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# python3 q1.py data.txt 


import math
import sys
import re
from pathlib import Path
from typing import Optional, Tuple
import csv
import time

import pandas as pd
import requests


IP_API_SINGLE_URL = "http://ip-api.com/json/{}"

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def geolocate_ip(ip: str) -> Optional[Tuple[float, float]]:
    delay = 1.0
    for attempt in range(10):
        url = IP_API_SINGLE_URL.format(ip)
        params = {"fields": "status,message,lat,lon,query"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 429:
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success":
                return None
            return float(data["lat"]), float(data["lon"])
        time.sleep(delay)
        delay *=2
    return None

_OK_RE = re.compile(
    r"address:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s*,\s*"
    r"min:\s*(?P<min>[\d.]+)ms\s*,\s*"
    r"max:\s*(?P<max>[\d.]+)ms\s*,\s*"
    r"avg:\s*(?P<avg>[\d.]+)ms"
)

def parse_txt_to_csv(input_txt: str) -> str:

    input_path = Path(input_txt)

    input_csv = input_path.with_suffix(".csv")

    kept = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as f, input_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["address", "min_ms", "max_ms", "avg_ms"])
        writer.writeheader()

        for line in f:
            m = _OK_RE.search(line)
            if not m:
                continue  

            writer.writerow({
                "address": m.group("ip"),
                "min_ms": float(m.group("min")),
                "max_ms": float(m.group("max")),
                "avg_ms": float(m.group("avg")),
            })
            kept += 1
    return input_csv

def main():

    input_txt = Path(sys.argv[1])
    input_csv = parse_txt_to_csv(input_txt)
    MY_LAT = 40.45
    MY_LON = -86.92

    output_csv = input_csv.with_name(f"{input_csv.stem}_with_distance.csv")

    df = pd.read_csv(input_csv)
    if df.shape[1] < 1:
        print("Error: CSV has no columns; first column must be destination IPs.")
        sys.exit(1)

    dest_col = df.columns[0]

    distances = []
    for raw in df[dest_col].astype(str):
        ip = raw.strip()
        if not ip:
            distances.append(None)
            continue

        loc = geolocate_ip(ip)
        if loc is None:
            distances.append(None)
            continue

        lat, lon = loc
        distances.append(haversine_km(MY_LAT, MY_LON, lat, lon))

    df["distance_km"] = distances
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()