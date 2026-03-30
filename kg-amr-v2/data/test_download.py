#!/usr/bin/env python3
"""Quick test: verify one BV-BRC genome download works before bulk run."""
import requests, time

session = requests.Session()
session.headers.update({"User-Agent": "KG-AMR-v2/1.0"})

test_id = "1328432.3"  # first E. coli genome in label file
urls = [
    ("ftp", f"https://ftp.bvbrc.org/genomes/{test_id}/{test_id}.fna"),
    ("api", f"https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{test_id})&http_accept=application/dna+fasta"),
]

for method, url in urls:
    print(f"Trying {method}: {url}")
    try:
        r = session.get(url, timeout=30)
        print(f"  Status: {r.status_code}, Bytes: {len(r.content)}")
        first = r.content[:100]
        print(f"  First 100 bytes: {repr(first)}")
        if r.status_code == 200 and b">" in r.content[:200]:
            print(f"  -> VALID FASTA confirmed. Download URLs work.")
            break
        else:
            print(f"  -> Not a valid FASTA response, trying fallback...")
    except Exception as e:
        print(f"  -> ERROR: {e}")
    time.sleep(1)
