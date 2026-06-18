"""
Sommel-AI :: Food pairing knowledge base
----------------------------------------
A curated mapping of wine variety -> (foods, flavor descriptors) drawn from:

  - Wine Folly / Court of Master Sommeliers / MasterClass general references
  - "Food & Wine Pairing Guide" by Antony Anderson (WSET Level 3, WSET4
    Diploma candidate) -- The Wine Matchmaker, 2020. Used for the expanded
    Italian, Thai, sparkling, and white-grape sections, the sauce-based pairing
    rules, and several additional varieties (Grenache, Montepulciano, Cava,
    Semillon, Marsanne).

Three layers of data are exposed:

  1. PAIRINGS              variety -> (foods, flavors)
  2. CUISINE_VARIETIES     cuisine -> list of varieties that traditionally pair
  3. PAIRING_PRINCIPLES    sauce / dish-element -> what wine attribute to seek
                           (e.g. tomato sauce -> low tannin; spicy heat -> off-dry)

The recommender uses (1) directly and consults (2) and (3) when the user
mentions a cuisine ("italian", "thai") or a sauce keyword ("tomato", "cream").
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# 1. Variety -> (foods, flavors)
# =============================================================================
# Each entry is (food keywords, flavor / descriptor keywords). Keys are
# lowercase. Multiple key spellings (rosé / rose, shiraz / syrah, pinot grigio
# / pinot gris) are kept as separate keys so dataset values match regardless
# of how the original review was written.
PAIRINGS: Dict[str, Tuple[List[str], List[str]]] = {

    # ---- light, aromatic whites ------------------------------------------------
    "marsanne": (
        ["seafood", "pork", "white fish", "halibut", "cod", "light meat", "antipasto"],
        ["pear", "melon", "blossom", "light-bodied", "honey", "acidic"],
    ),
    "semillon": (
        ["seafood", "pork", "soft cheese", "crab cakes", "ceviche", "prawns"],
        ["citrus", "green apple", "waxy", "lanolin", "honeyed", "high acid"],
    ),
    "pinot grigio": (
        ["sushi", "light pasta", "shellfish", "salad", "antipasti", "antipasto",
         "pasta carbonara", "italian", "olive oil pasta", "seafood pasta",
         "prawns", "halibut", "cod", "white fish", "salami", "soppressata"],
        ["light", "crisp", "citrus", "pear", "clean", "refreshing", "melon",
         "high acid", "fresh"],
    ),
    "pinot gris": (
        ["pork", "chicken", "white fish", "halibut", "cod", "antipasto",
         "soft cheese", "richer dishes", "salmon", "thai", "spicy asian",
         "cured meats", "ham", "soppressata", "canned fish", "sardines",
         "anchovies"],
        ["ripe pear", "white peach", "lemon zest", "apricot", "honey",
         "meyer lemon", "golden apple", "medium-bodied", "balanced"],
    ),
    "sauvignon blanc": (
        ["salad", "goat cheese", "ceviche", "sushi", "shellfish", "herbs",
         "chilean sea bass", "sablefish", "fatty fish", "crab cakes", "brie",
         "soft cheese", "cheesecake", "crème brûlée", "cupcakes",
         "tropical fruit desserts", "asparagus", "green salad"],
        ["crisp", "citrus", "grassy", "herbaceous", "zesty", "grapefruit",
         "passionfruit", "gooseberry", "tropical", "pineapple", "mango",
         "melon", "mandarin", "high acid"],
    ),
    "riesling": (
        ["thai", "spicy food", "pork", "duck", "asian cuisine", "curry",
         "prawns", "seafood", "smoked fish", "indian", "vietnamese",
         "spicy asian", "chinese", "ginger", "lime", "chili",
         "shrimp", "cheese"],
        ["sweet", "off-dry", "floral", "stone fruit", "peach", "honey",
         "petrol", "kerosene", "lemon", "lime", "grapefruit", "mineral",
         "wet stone", "residual sugar", "high acid", "crisp"],
    ),
    "gewürztraminer": (
        ["thai", "indian", "spicy food", "munster cheese", "curry", "moroccan",
         "vietnamese", "ginger", "lychee", "asian cuisine", "spiced pork"],
        ["floral", "lychee", "spicy", "rose", "aromatic", "off-dry", "ginger",
         "residual sugar"],
    ),
    "gewurztraminer": (   # ascii alias, same data
        ["thai", "indian", "spicy food", "munster cheese", "curry", "moroccan",
         "vietnamese", "ginger", "lychee", "asian cuisine", "spiced pork"],
        ["floral", "lychee", "spicy", "rose", "aromatic", "off-dry", "ginger",
         "residual sugar"],
    ),
    "viognier": (
        ["roast chicken", "creamy pasta", "indian", "thai", "soft cheese"],
        ["floral", "peach", "apricot", "rich", "aromatic"],
    ),
    "chenin blanc": (
        ["pork", "roast chicken", "salad", "soft cheese", "thai",
         "spicy asian", "ham"],
        ["honey", "apple", "off-dry", "versatile", "floral"],
    ),

    # ---- richer whites ---------------------------------------------------------
    "chardonnay": (
        ["lobster", "crab", "shrimp", "scallops", "shellfish", "lobster tails",
         "shrimp scampi", "crab cakes", "roast chicken", "chicken alfredo",
         "creamy pasta", "risotto", "pot pies", "brie", "cheesy dishes",
         "chicken breast", "apple desserts", "pear desserts", "pear tarts",
         "fruit pies", "crumbles"],
        ["buttery", "oaky", "vanilla", "rich", "tropical", "apple", "lush",
         "golden apple", "pear", "lemon", "toasty", "mouthfeel", "medium-bodied"],
    ),
    "white blend": (
        ["seafood", "salad", "chicken", "light pasta", "antipasto"],
        ["balanced", "versatile", "medium-bodied"],
    ),

    # ---- light reds ------------------------------------------------------------
    "pinot noir": (
        ["salmon", "duck", "mushroom", "roast chicken", "pork", "lamb", "turkey",
         "grilled salmon", "spicy dishes", "asian cuisine", "sushi", "ramen",
         "general tso's chicken", "beef with broccoli", "tacos", "burgers",
         "chili", "stuffed mushrooms", "truffle", "mushroom sauce",
         "flank steak", "sirloin", "veal", "pulled pork", "cured meats",
         "sharp cheeses", "aged cheddar", "gruyere", "roasted vegetables",
         "game birds", "lean meats", "chicken scarpariello", "fra diavolo",
         "tonkotsu ramen", "italian sub", "turkey club", "pastrami reuben"],
        ["earthy", "red fruit", "cherry", "raspberry", "silky", "light",
         "elegant", "medium-bodied", "strawberry", "mushroom", "forest floor",
         "dusty", "oregon"],
    ),

    # ---- low-tannin reds for tomato / spicy food ------------------------------
    "grenache": (
        ["thai", "curry", "spicy food", "tomato sauce", "tomato pasta",
         "pizza", "vietnamese", "indian", "spicy asian", "mediterranean",
         "rotisserie chicken", "grilled vegetables", "moroccan tagine"],
        ["fruity", "low tannin", "red berry", "strawberry", "raspberry",
         "soft", "warm-climate", "spicy"],
    ),
    "sangiovese": (
        ["pizza", "pasta", "tomato sauce", "italian", "grilled vegetables",
         "spaghetti", "lasagna", "chianti", "marinara", "bolognese",
         "tomato pasta", "ragu", "salumi"],
        ["bright", "cherry", "herbal", "acidic", "tomato-friendly",
         "red fruit", "medium-bodied"],
    ),
    "montepulciano": (
        ["pizza", "pasta", "tomato sauce", "italian", "ragu", "lasagna",
         "spaghetti", "meatballs", "marinara"],
        ["cherry", "red berries", "low tannin", "fruity", "medium-bodied",
         "approachable"],
    ),

    # ---- medium / structured reds ---------------------------------------------
    "merlot": (
        ["roast chicken", "pork", "beef stew", "pasta", "mushroom", "duck",
         "young cheddar", "gouda", "mild cheese", "grilled tuna", "tuna steaks",
         "miso salmon", "charcuterie", "salami", "soup", "ham", "spiral ham",
         "honey ham"],
        ["soft", "plum", "chocolate", "round", "medium-bodied", "smooth",
         "blueberry", "approachable", "vanilla", "oak"],
    ),
    "tempranillo": (
        ["lamb", "tapas", "chorizo", "paella", "hard cheese", "grilled meats",
         "spanish", "manchego", "rioja"],
        ["leather", "tobacco", "cherry", "earthy", "spanish", "structured"],
    ),
    "red blend": (
        ["steak", "pasta", "burgers", "roast", "grilled meats", "lamb",
         "swordfish", "veal chops", "bbq pork", "pork belly", "roasted vegetables",
         "coq au vin", "chicken mole", "teriyaki", "chimichurri",
         "braised short ribs", "bbq ribs", "carne asada", "grilled sausage"],
        ["versatile", "balanced", "medium-bodied", "fruit-forward",
         "blackberry", "lavender", "rosemary", "silky tannin", "complex"],
    ),
    "bordeaux-style red blend": (
        ["steak", "lamb", "roast", "hard cheese", "duck", "lamb kebabs",
         "blackened swordfish", "veal", "bbq pork buns", "venison"],
        ["structured", "tannic", "blackcurrant", "oak", "elegant", "plum",
         "cedar", "vanilla", "dark fruit"],
    ),
    "portuguese red": (
        ["grilled meats", "stew", "lamb", "sausage"],
        ["earthy", "dark fruit", "rustic", "structured"],
    ),

    # ---- bold / tannic reds for fatty meat ------------------------------------
    "cabernet sauvignon": (
        ["steak", "ribeye", "lamb", "burger", "hard cheese", "aged cheddar",
         "bbq", "tomahawk", "porterhouse", "dry-aged ribeye", "strip steak",
         "crown pork", "london broil", "prime rib", "large roasts",
         "hearty stews", "beef stew", "venison stew", "grilled vegetables",
         "dark chocolate", "chocolate truffles", "blue cheese"],
        ["bold", "tannic", "blackcurrant", "cassis", "structured", "powerful",
         "oak", "full-bodied", "rich tannin", "dark fruit", "complex",
         "herbaceous", "spice"],
    ),
    "syrah": (
        ["bbq", "ribs", "lamb", "venison", "grilled meats", "stew",
         "steak", "shiraz steak", "braised meats", "moroccan", "duck"],
        ["smoky", "peppery", "spicy", "dark fruit", "full-bodied", "blackberry",
         "leather"],
    ),
    "shiraz": (
        ["bbq", "ribs", "grilled meats", "stew", "burgers", "steak",
         "lamb roast", "beef bourguignon", "dark chocolate", "hard cheese",
         "dried fruit"],
        ["jammy", "peppery", "bold", "dark fruit", "ripe", "full-bodied",
         "smoky", "blackberry"],
    ),
    "zinfandel": (
        ["bbq", "pizza", "burgers", "pulled pork", "spiced meats", "ribs"],
        ["jammy", "spicy", "bold", "blackberry", "fruit-forward"],
    ),
    "malbec": (
        ["steak", "grilled beef", "lamb", "chimichurri", "empanadas",
         "argentine asado", "carne asada"],
        ["bold", "plum", "dark fruit", "smoky", "structured"],
    ),
    "nebbiolo": (
        ["truffle", "braised beef", "risotto", "lamb", "aged cheese",
         "osso buco", "meat ragu", "bistecca", "fiorentina", "fatty meat",
         "italian roast", "barolo pairing", "barbaresco pairing",
         "porcini", "wild boar"],
        ["tannic", "rose", "tar", "cherry", "earthy", "long-aging",
         "high tannin", "structured", "piedmont"],
    ),

    # ---- rosé / sparkling / dessert -------------------------------------------
    "rosé": (
        ["salad", "grilled vegetables", "salmon", "summer food", "charcuterie",
         "smoked salmon", "arctic char", "tuna", "mushroom risotto",
         "chicken marsala", "veal marsala", "curry mussels", "kebabs",
         "stuffed olives", "pickled vegetables", "spicy mustard"],
        ["light", "fresh", "strawberry", "watermelon", "summer", "dry",
         "raspberry", "white cherry", "bright"],
    ),
    "rose": (   # ascii alias
        ["salad", "grilled vegetables", "salmon", "summer food", "charcuterie"],
        ["light", "fresh", "strawberry", "watermelon", "summer", "dry"],
    ),
    "champagne blend": (
        ["oysters", "caviar", "fried food", "fried chicken", "popcorn",
         "brunch", "sushi", "smoked salmon", "lobster", "dungeness crab",
         "shrimp scampi", "almond croissants", "english muffins",
         "blueberry muffins", "blinis", "baked pastry", "appetizers",
         "salads", "birthday cake", "fruit tarts", "chocolate-covered fruit"],
        ["sparkling", "crisp", "bright", "yeasty", "citrus", "celebration",
         "lemon", "orange", "apple", "pear", "stone fruit", "clean"],
    ),
    "sparkling blend": (
        ["oysters", "fried food", "popcorn", "brunch", "sushi", "appetizers",
         "shellfish", "prawns", "thai prawns", "tapas", "paella",
         "artichokes", "asparagus", "salami", "italian hard cheese",
         "asiago", "provolone", "charcuterie", "crusty bread"],
        ["sparkling", "bright", "crisp", "fizzy", "citrus", "high acid",
         "herbal", "floral", "yeasty"],
    ),
    "prosecco": (
        ["brunch", "appetizers", "fruit", "light desserts", "salami",
         "charcuterie", "hard italian cheese", "asiago", "provolone",
         "crusty bread", "antipasto"],
        ["sparkling", "fruity", "light", "off-dry", "approachable",
         "italian", "slightly sweet"],
    ),
    "moscato": (
        ["salty snacks", "popcorn", "nuts", "fresh cheese", "goat cheese",
         "mozzarella", "burrata", "blue cheese", "cold soup", "spicy soup",
         "crudité", "birthday cake", "cheesecake", "baklava", "ice cream",
         "sorbet", "strawberries"],
        ["sweet", "light-bodied", "apricot", "nectarine", "grapefruit",
         "honeydew", "mandarin", "off-dry", "aromatic"],
    ),
    "port": (
        ["chocolate", "blue cheese", "nuts", "dessert", "dark chocolate",
         "stilton"],
        ["sweet", "rich", "fortified", "dark fruit", "dessert"],
    ),
}


# =============================================================================
# 2. Cuisine -> list of well-pairing varieties
# =============================================================================
# Used by the recommender when a user mentions a cuisine in their food box.
CUISINE_VARIETIES: Dict[str, List[str]] = {
    "italian": [
        "sangiovese", "montepulciano", "nebbiolo", "pinot grigio",
        "prosecco", "sparkling blend",
    ],
    "thai": [
        "riesling", "gewürztraminer", "pinot gris", "grenache", "chenin blanc",
    ],
    "indian": [
        "riesling", "gewürztraminer", "viognier", "grenache",
    ],
    "japanese": [
        "pinot noir", "riesling", "champagne blend", "sparkling blend",
    ],
    "chinese": [
        "riesling", "pinot noir", "gewürztraminer",
    ],
    "mexican": [
        "grenache", "zinfandel", "pinot noir", "riesling",
    ],
    "french": [
        "pinot noir", "chardonnay", "bordeaux-style red blend",
        "sauvignon blanc", "champagne blend",
    ],
    "spanish": [
        "tempranillo", "sparkling blend", "grenache",
    ],
    "bbq": [
        "shiraz", "syrah", "zinfandel", "malbec", "red blend",
    ],
    "vegetarian": [
        "pinot noir", "rosé", "sauvignon blanc", "grenache",
    ],
}


# =============================================================================
# 3. Pairing principles (sauce / dish-element -> wine attribute)
# =============================================================================
# Captures the rules-of-thumb from the Wine Matchmaker guide so the recommender
# can reason about *why* certain matches work, not just memorized lists.
PAIRING_PRINCIPLES: Dict[str, str] = {
    # fat / acid / tannin axis
    "fatty":         "tannin or high acid to cut through the fat",
    "salty":         "off-dry wine or dry sherry; high-tannin reds will clash",
    "spicy":         "off-dry whites (Riesling, Gewürztraminer) or low-tannin "
                     "fruity reds (Grenache); high-tannin reds clash with chili",
    "creamy":        "rich whites with body and oak (Chardonnay) or off-dry",
    "tomato":        "low-tannin reds (Grenache, Sangiovese, Montepulciano); "
                     "high-tannin reds turn metallic against tomato acid",
    "smoky":         "Syrah, Shiraz, Malbec, smoky reds",
    "sweet dessert": "wine must be sweeter than the dessert (Moscato, Port, "
                     "late-harvest, botrytis)",
    "raw seafood":   "high-acid crisp whites or sparkling (Riesling, Sauvignon "
                     "Blanc, Champagne)",
    "grilled meat":  "tannic full-bodied reds (Cabernet, Syrah, Shiraz, "
                     "Nebbiolo, Malbec)",
    "soft cheese":   "Sauvignon Blanc, Champagne, Pinot Gris",
    "hard cheese":   "tannic reds (Cabernet, Nebbiolo, aged Pinot Noir)",
}


# =============================================================================
# Public helpers
# =============================================================================
def keywords_for(variety: str | None) -> Tuple[List[str], List[str]]:
    """Return (foods, flavors) for a variety. Empty lists if unknown."""
    if not variety:
        return [], []
    return PAIRINGS.get(variety.strip().lower(), ([], []))


def varieties_for_food(food: str) -> List[str]:
    """
    Reverse lookup: which varieties pair well with a given food keyword.
    Matches both literal food keywords and cuisine names.
    """
    if not food:
        return []
    food = food.lower().strip()

    # Cuisine-level match first (cheap, high-quality signal)
    if food in CUISINE_VARIETIES:
        return list(CUISINE_VARIETIES[food])

    # Otherwise scan the PAIRINGS food lists
    out = []
    for variety, (foods, _flavors) in PAIRINGS.items():
        if any(food in f or f in food for f in foods):
            out.append(variety)

    # Also pull cuisine matches by partial substring (e.g. "thai curry" -> thai)
    for cuisine, varieties in CUISINE_VARIETIES.items():
        if cuisine in food:
            out.extend(v for v in varieties if v not in out)

    return out


def food_text(variety: str | None) -> str:
    """Concat food + flavor keywords into a single string, for boosting TF-IDF queries."""
    foods, flavors = keywords_for(variety)
    return " ".join(foods + flavors)


def principle_for(food_or_attribute: str) -> str | None:
    """
    Return the pairing principle (rule of thumb) that applies to a given food
    or dish attribute. Used by the UI to show *why* a match was made.

    Examples:
        principle_for("spicy thai curry") -> off-dry whites or low-tannin reds...
        principle_for("ribeye steak")     -> tannic full-bodied reds...
    """
    if not food_or_attribute:
        return None
    s = food_or_attribute.lower()
    for key, principle in PAIRING_PRINCIPLES.items():
        if key in s:
            return principle
    # also try the inverse direction (food -> principle key)
    if any(w in s for w in ("steak", "ribeye", "grill", "bbq", "lamb")):
        return PAIRING_PRINCIPLES["grilled meat"]
    if any(w in s for w in ("oyster", "sushi", "ceviche", "sashimi")):
        return PAIRING_PRINCIPLES["raw seafood"]
    if any(w in s for w in ("curry", "chili", "thai", "indian", "szechuan")):
        return PAIRING_PRINCIPLES["spicy"]
    if any(w in s for w in ("brie", "camembert", "goat cheese")):
        return PAIRING_PRINCIPLES["soft cheese"]
    if any(w in s for w in ("cheddar", "parmesan", "aged")):
        return PAIRING_PRINCIPLES["hard cheese"]
    return None
