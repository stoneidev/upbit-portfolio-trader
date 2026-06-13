import os
import sys
import json
import time
import subprocess

# Ensure playwright is installed
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not found. Installing playwright...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    print("Installing chromium browser binaries...")
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.sync_api import sync_playwright

def extract_seo_metadata(page):
    # Wait for page to settle
    page.wait_for_timeout(3000)
    
    # Evaluate metadata extraction safely in browser DOM context
    metadata = page.evaluate("""() => {
        const getMetaContent = (name) => {
            const el = document.querySelector(`meta[name="${name}"]`) || document.querySelector(`meta[property="${name}"]`) || document.querySelector(`meta[name="${name.toLowerCase()}"]`) || document.querySelector(`meta[property="${name.toLowerCase()}"]`);
            return el ? el.getAttribute('content') : '';
        };
        
        const getCanonical = () => {
            const el = document.querySelector('link[rel="canonical"]');
            return el ? el.getAttribute('href') : '';
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
            description: getMetaContent('description') || getMetaContent('Description') || '',
            canonical: getCanonical() || '',
            og_tags: {
                'og:title': getMetaContent('og:title') || '',
                'og:description': getMetaContent('og:description') || '',
                'og:image': getMetaContent('og:image') || '',
                'og:url': getMetaContent('og:url') || ''
            },
            structured_data: getStructuredData()
        };
    }""")
    
    metadata["url"] = page.url
    return metadata

def main():
    homepage_url = "https://stylekorean.com?device=mobile"
    screenshot_path = "/Users/stoni/.gemini/antigravity/brain/804626a0-d8b9-45cd-b221-8f491a80d3c3/stylekorean_product_detail.png"
    
    # Ensure parent directory of screenshot exists
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    
    results = {}
    
    with sync_playwright() as p:
        # Emulate a mobile device (iPhone 12 is a good representative mobile viewport)
        device = p.devices["iPhone 12"]
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()
        
        print(f"Navigating to homepage: {homepage_url}...")
        try:
            page.goto(homepage_url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"Warning during page load: {e}")
        
        # Additional wait for AJAX/scripts
        page.wait_for_timeout(5000)
        
        print("Extracting homepage SEO metadata...")
        results["homepage"] = extract_seo_metadata(page)
        
        # Find product links
        print("Finding product links...")
        product_href = None
        
        # Try document-wide search for product detail URL in DOM
        try:
            product_href = page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a'));
                const prodLink = links.find(a => {
                    const href = a.getAttribute('href') || '';
                    return href.includes('goods_detail.php') || href.includes('/products/') || href.includes('goods_no=');
                });
                if (prodLink) {
                    let href = prodLink.getAttribute('href');
                    if (href.startsWith('/')) {
                        return 'https://stylekorean.com' + href;
                    }
                    return href;
                }
                return null;
            }""")
            if product_href:
                print(f"Found product link from DOM evaluation: {product_href}")
        except Exception as e:
            print(f"Error during DOM evaluation: {e}")
            
        # Fallback if no link found
        if not product_href:
            print("No product URL found. Looking at sample of links on page...")
            try:
                links_sample = page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a')).map(a => a.href).filter(h => h && h.startsWith('http')).slice(0, 15);
                }""")
                for l in links_sample:
                    if 'goods' in l or 'product' in l or 'detail' in l:
                        product_href = l
                        print(f"Found fallback product link from sample: {product_href}")
                        break
            except Exception:
                pass
                
        if not product_href:
            print("Warning: Could not find any product detail link to navigate. Using a placeholder or popular product link to continue.")
            product_href = "https://stylekorean.com/shop/goods_detail.php?goods_no=15998"
            
        print(f"Navigating to product detail page: {product_href}...")
        try:
            page.goto(product_href, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"Warning during product page load: {e}")
            
        page.wait_for_timeout(5000)
        
        print("Extracting product page SEO metadata...")
        results["product_page"] = extract_seo_metadata(page)
        
        print(f"Saving screenshot to {screenshot_path}...")
        page.screenshot(path=screenshot_path, full_page=False)
        print("Screenshot saved successfully.")
        
        browser.close()
        
    # Output the result JSON to stdout with a marker for parsing
    print("AUDIT_RESULTS_START")
    print(json.dumps(results, indent=2))
    print("AUDIT_RESULTS_END")

if __name__ == "__main__":
    main()
