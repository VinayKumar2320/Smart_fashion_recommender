# recommender.py
from fashion_api import get_pinterest_suggestions
from pinterest_viewer import display_pinterest_images_verified  # ✅ updated import

def suggest_outfit(cloth_type, color_name):
    """
    Suggest outfit ideas based on detected clothing type and color.
    Returns a text suggestion and a list of Pinterest image URLs.
    """
    print(f"[DEBUG] Querying for: {color_name} {cloth_type.replace('_', ' ')} outfit ideas")

    # Create query
    query = f"{color_name} {cloth_type.replace('_', ' ')} outfit ideas"
    image_urls = get_pinterest_suggestions(query, max_results=10)

    # Simple text-based fashion advice
    base_color = color_name.lower()
    if "blue" in base_color:
        suggestion = f"Since you're wearing a {color_name.lower()} {cloth_type.replace('_', ' ')}, try pairing it with something white or beige."
    elif "white" in base_color:
        suggestion = f"A {color_name.lower()} {cloth_type.replace('_', ' ')} goes well with denim or pastel shades."
    elif "black" in base_color:
        suggestion = f"A {color_name.lower()} {cloth_type.replace('_', ' ')} pairs well with bright or neutral bottoms."
    else:
        suggestion = f"Try combining your {color_name.lower()} {cloth_type.replace('_', ' ')} with something complementary!"

    return suggestion, image_urls


# --------------------------
# Run test
if __name__ == "__main__":
    cloth_type = "short_sleeved_shirt"
    color_name = "Light Blue"

    suggestion, urls = suggest_outfit(cloth_type, color_name)

    print(suggestion)
    for u in urls:
        print(u)

    # Display verified outfit images in a window ✅
    if urls:
        display_pinterest_images_verified(urls, color_name)
