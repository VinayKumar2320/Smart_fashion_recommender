import requests
import re

def get_pinterest_suggestions(query, max_results=10):
    """
    Fetch Pinterest-style images using Google first, then Bing as fallback.
    Returns clean Pinterest CDN URLs.
    """
    def fetch_google(query):
        search_query = query.replace(" ", "+") + "+outfit+fashion+street+style+clothes"
        url = f"https://www.google.com/search?tbm=isch&q=site:pinterest.com+{search_query}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[WARN] Google fetch failed: {e}")
            return []
        return re.findall(r"https://i\.pinimg\.com/[^\s\"'>]+?\.(?:jpg|jpeg|png)", r.text)

    def fetch_bing(query):
        search_query = query.replace(" ", "+") + "+outfit+fashion+style+streetwear+clothes"
        url = f"https://www.bing.com/images/search?q=site:pinterest.com+{search_query}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[WARN] Bing fetch failed: {e}")
            return []
        return re.findall(r"https://i\.pinimg\.com/[^\s\"'>]+?\.(?:jpg|jpeg|png)", r.text)

    print(f"[DEBUG] Querying for Pinterest images: {query}")

    # Try Google first
    images = fetch_google(query)
    if len(images) < max_results:
        print("[DEBUG] Using Bing fallback")
        images = fetch_bing(query)

    # Deduplicate and limit
    images = list(dict.fromkeys(images))[:max_results]

    if not images:
        print("⚠️ No Pinterest-style images found. Try a broader search.")

    return images


# -------------------------------
# Example test
if __name__ == "__main__":
    query = "light blue short sleeved shirt"
    results = get_pinterest_suggestions(query, max_results=8)

    print(f"\nFound {len(results)} Pinterest-style outfit image links:\n")
    for url in results:
        print(url)
