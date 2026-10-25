import urllib.request
import re

url = "https://earthobservatory.nasa.gov/world-of-change/Deforestation"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    links = set(re.findall(r'https://assets\.science\.nasa\.gov/[^\s"\']+\.jpg', html))
    for l in sorted(links):
        print(l)
except Exception as e:
    print("Error:", e)
