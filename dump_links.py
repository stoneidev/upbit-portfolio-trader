from playwright.sync_api import sync_playwright
import json

def main():
    url = "https://stylekorean.com/?device=mobile"
    with sync_playwright() as p:
        device = p.devices["iPhone 12"]
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()
        
        print("Navigating...")
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(5000) # Wait extra time
        
        # Dump all href attributes
        hrefs = page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a')).map(a => ({
                text: a.innerText.trim(),
                href: a.getAttribute('href')
            })).filter(item => item.href);
        }""")
        
        print("--- EXTRACTED LINKS (First 40) ---")
        for i, item in enumerate(hrefs[:40]):
            print(f"{i}: [{item['text']}] -> {item['href']}")
            
        browser.close()

if __name__ == "__main__":
    main()
