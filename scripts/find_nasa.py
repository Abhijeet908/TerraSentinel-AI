import urllib.request
import re

url = 'https://earthobservatory.nasa.gov/images/150111/lake-mead-keeps-dropping'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    # Find all jpg links
    links = re.findall(r'https://assets\.science\.nasa\.gov/[^\s"\']+\.jpg', html)
    print("Found links:", links)
except Exception as e:
    print("Error:", e)
