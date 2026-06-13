import urllib.request
import re

url = "https://stylekorean.com/?device=mobile"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'}
)

try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8', errors='ignore')
        
        # Extract <head> section
        head_match = re.search(r'<head>(.*?)</head>', html, re.DOTALL | re.IGNORECASE)
        if head_match:
            head_content = head_match.group(1)
            print("--- RAW HEAD TAGS ---")
            for line in head_content.split('\n'):
                if any(x in line for x in ['meta', 'title', 'link', 'script type="application/ld+json"']):
                    print(line.strip())
        else:
            print("Could not find <head> tags")
            print("First 500 chars of HTML:", html[:500])
except Exception as e:
    print("Error:", e)
