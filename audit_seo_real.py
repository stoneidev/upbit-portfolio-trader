import os
import json
from playwright.sync_api import sync_playwright

def extract_seo_metadata(page):
    # Wait for page elements to load
    page.wait_for_timeout(3000)
    
    metadata = page.evaluate("""() => {
        const getMetaContent = (name) => {
            const el = document.querySelector(`meta[name="${name}"]`) || 
                       document.querySelector(`meta[property="${name}"]`) || 
                       document.querySelector(`meta[name="${name.toLowerCase()}"]`) || 
                       document.querySelector(`meta[property="${name.toLowerCase()}"]`);
            return el ? el.getAttribute('content') : '';
        };
        
        const getCanonical = () => {
            const el = document.querySelector('link[rel="canonical"]');
            return el ? el.getAttribute('href') : '';
        };
        
        const getHeadings = () => {
            const headings = {};
            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].forEach(tag => {
                headings[tag] = Array.from(document.querySelectorAll(tag))
                                     .map(el => el.innerText.trim())
                                     .filter(t => t.length > 0);
            });
            return headings;
        };
        
        const getStructuredData = () => {
            const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
            const data = [];
            scripts.forEach(script => {
                try {
                    const text = script.innerText || script.textContent || '';
                    if (text.trim()) {
                        data.push(JSON.parse(text.trim()));
                    }
                } catch (e) {
                    data.push({ error: "JSON parse error: " + e.message, raw: script.textContent });
                }
            });
            return data;
        };
        
        return {
            title: document.title || '',
            description: getMetaContent('description') || '',
            canonical: getCanonical() || '',
            og_tags: {
                'og:title': getMetaContent('og:title') || '',
                'og:description': getMetaContent('og:description') || '',
                'og:image': getMetaContent('og:image') || '',
                'og:url': getMetaContent('og:url') || '',
                'og:type': getMetaContent('og:type') || ''
            },
            headings: getHeadings(),
            structured_data: getStructuredData()
        };
    }""")
    
    metadata["url"] = page.url
    return metadata

def main():
    homepage_url = "https://stylekorean.com/?device=mobile"
    product_url = "https://www.stylekorean.com/shop/unpa-bubi-bubi-bubble-lip-scrub-red-lip-remover/1776732734/?device=mobile"
    screenshot_path = "/Users/stoni/.gemini/antigravity/brain/804626a0-d8b9-45cd-b221-8f491a80d3c3/stylekorean_product_detail.png"
    
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    results = {}
    
    with sync_playwright() as p:
        device = p.devices["iPhone 12"]
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()
        
        print(f"Navigating to homepage: {homepage_url}...")
        try:
            page.goto(homepage_url, wait_until="networkidle", timeout=60000)
        except Exception as e:
            print(f"Homepage load warning: {e}")
        page.wait_for_timeout(3000)
        results["homepage"] = extract_seo_metadata(page)
        
        print(f"Navigating to product page: {product_url}...")
        try:
            page.goto(product_url, wait_until="networkidle", timeout=60000)
        except Exception as e:
            print(f"Product page load warning: {e}")
        page.wait_for_timeout(3000)
        results["product_page"] = extract_seo_metadata(page)
        
        print(f"Saving screenshot to {screenshot_path}...")
        page.screenshot(path=screenshot_path, full_page=False)
        print("Screenshot saved.")
        
        browser.close()
        
    print("AUDIT_RESULTS_START")
    print(json.dumps(results, indent=2))
    print("AUDIT_RESULTS_END")

if __name__ == "__main__":
    main()
