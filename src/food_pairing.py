"""
Sommel-AI :: Food pairing knowledge base
----------------------------------------
Curated mapping of wine variety -> (foods, flavor descriptors), drawn from
multiple professional sources:

  - Wine Folly / Court of Master Sommeliers / MasterClass general references
  - "Food & Wine Pairing Guide" by Antony Anderson (WSET Level 3) -- The Wine
    Matchmaker, 2020. Italian, Thai, sparkling, and white-grape sections.
  - "Court of Master Sommeliers Europe -- Food and Wine Matching" principles
    document. Used for the structural pairing rules (weight, acid, tannin, etc).
  - Thermador "The Ultimate Wine Pairing Guide" -- the detailed per-variety
    cheese / meat / seafood / veggie / herb / sauce / dessert grid.
  - H-E-B "Wine Pairing Chart" -- the wine x food category matrix.

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

    # =========================================================================
    # LIGHT, AROMATIC WHITES
    # =========================================================================
    "marsanne": (
        ["seafood", "pork", "white fish", "halibut", "cod", "light meat", "antipasto"],
        ["pear", "melon", "blossom", "light-bodied", "honey", "acidic"],
    ),
    "semillon": (
        ["seafood", "pork", "soft cheese", "crab cakes", "ceviche", "prawns",
         "asparagus"],
        ["citrus", "green apple", "waxy", "lanolin", "honeyed", "high acid"],
    ),
    "pinot grigio": (
        ["sushi", "light pasta", "shellfish", "salad", "antipasti", "antipasto",
         "pasta carbonara", "italian", "olive oil pasta", "seafood pasta",
         "prawns", "halibut", "cod", "white fish", "salami", "soppressata",
         # H-E-B chart adds:
         "bbq pork", "salmon", "shrimp", "soft rich cheese"],
        ["light", "crisp", "citrus", "pear", "clean", "refreshing", "melon",
         "high acid", "fresh"],
    ),
    "pinot gris": (
        ["pork", "chicken", "white fish", "halibut", "cod", "antipasto",
         "soft cheese", "richer dishes", "salmon", "thai", "spicy asian",
         "cured meats", "ham", "soppressata", "canned fish", "sardines",
         "anchovies", "bbq pork", "shrimp", "soft rich cheese"],
        ["ripe pear", "white peach", "lemon zest", "apricot", "honey",
         "meyer lemon", "golden apple", "medium-bodied", "balanced"],
    ),
    "sauvignon blanc": (
        ["salad", "goat cheese", "ceviche", "sushi", "shellfish", "herbs",
         "chilean sea bass", "sablefish", "fatty fish", "crab cakes", "brie",
         "soft cheese", "cheesecake", "crème brûlée", "cupcakes",
         "tropical fruit desserts", "asparagus", "green salad",
         # Thermador additions:
         "asiago", "gruyere", "pine nuts", "chicken", "turkey", "pork",
         "oysters", "scallops", "smoked salmon", "citrus", "artichoke",
         "melon", "chives", "tarragon", "garlic", "basil", "anchovy dressing",
         "olive oil", "sorbet", "key lime pie", "apricot torte",
         # H-E-B additions:
         "fish", "pasta", "tomato veggies", "japanese cuisine", "japan"],
        ["crisp", "citrus", "grassy", "herbaceous", "zesty", "grapefruit",
         "passionfruit", "gooseberry", "tropical", "pineapple", "mango",
         "melon", "mandarin", "high acid"],
    ),
    "riesling": (
        ["thai", "spicy food", "pork", "duck", "asian cuisine", "curry",
         "prawns", "seafood", "smoked fish", "indian", "vietnamese",
         "spicy asian", "chinese", "ginger", "lime", "chili", "shrimp",
         "cheese",
         # Thermador additions:
         "pecans", "pepper jack", "smoked turkey", "foie gras", "sea bass",
         "oysters", "poached fish", "apricots", "chili peppers",
         "sweet potato", "bell peppers", "rosemary", "lemongrass", "sage",
         "sweet bbq", "black bean sauce", "honey mustard", "apple pie",
         "caramel sauce", "baked pears",
         # H-E-B additions:
         "beef", "poultry"],
        ["sweet", "off-dry", "floral", "stone fruit", "peach", "honey",
         "petrol", "kerosene", "lemon", "lime", "grapefruit", "mineral",
         "wet stone", "residual sugar", "high acid", "crisp"],
    ),
    "gewürztraminer": (
        ["thai", "indian", "spicy food", "munster cheese", "curry", "moroccan",
         "vietnamese", "ginger", "lychee", "asian cuisine", "spiced pork",
         # H-E-B / Thermador additions:
         "beef", "poultry", "shrimp", "asian/indian", "spicy",
         "smoked salmon", "turkey", "duck"],
        ["floral", "lychee", "spicy", "rose", "aromatic", "off-dry", "ginger",
         "residual sugar"],
    ),
    "gewurztraminer": (   # ascii alias, same data
        ["thai", "indian", "spicy food", "munster cheese", "curry", "moroccan",
         "vietnamese", "ginger", "lychee", "asian cuisine", "spiced pork",
         "beef", "poultry", "shrimp", "asian/indian", "spicy"],
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

    # =========================================================================
    # RICHER WHITES
    # =========================================================================
    "chardonnay": (
        ["lobster", "crab", "shrimp", "scallops", "shellfish", "lobster tails",
         "shrimp scampi", "crab cakes", "roast chicken", "chicken alfredo",
         "creamy pasta", "risotto", "pot pies", "brie", "cheesy dishes",
         "chicken breast", "apple desserts", "pear desserts", "pear tarts",
         "fruit pies", "crumbles",
         # Thermador additions:
         "almonds", "veal", "pork loin", "halibut", "smoked trout", "mango",
         "tangerines", "bok choy", "sesame", "saffron", "rosemary", "pesto",
         "horseradish", "tropical salsa", "banana bread", "pecan pie",
         "pound cake",
         # H-E-B additions:
         "simple beef", "fish", "cream sauce pasta", "green veggies",
         "soft rich cheese", "japanese cuisine", "japan", "asian/indian"],
        ["buttery", "oaky", "vanilla", "rich", "tropical", "apple", "lush",
         "golden apple", "pear", "lemon", "toasty", "mouthfeel",
         "medium-bodied"],
    ),
    "chablis": (   # unoaked Chardonnay from Burgundy
        ["oysters", "shellfish", "raw seafood", "green vegetables", "salad",
         "goat cheese", "crab", "scallops"],
        ["crisp", "mineral", "unoaked", "citrus", "lean", "flinty",
         "high acid"],
    ),
    "white blend": (
        ["seafood", "salad", "chicken", "light pasta", "antipasto"],
        ["balanced", "versatile", "medium-bodied"],
    ),

    # =========================================================================
    # LIGHT REDS
    # =========================================================================
    "pinot noir": (
        ["salmon", "duck", "mushroom", "roast chicken", "pork", "lamb",
         "turkey", "grilled salmon", "spicy dishes", "asian cuisine", "sushi",
         "ramen", "general tso's chicken", "beef with broccoli", "tacos",
         "burgers", "chili", "stuffed mushrooms", "truffle", "mushroom sauce",
         "flank steak", "sirloin", "veal", "pulled pork", "cured meats",
         "sharp cheeses", "aged cheddar", "gruyere", "roasted vegetables",
         "game birds", "lean meats", "chicken scarpariello", "fra diavolo",
         "tonkotsu ramen", "italian sub", "turkey club", "pastrami reuben",
         # Thermador additions:
         "walnuts", "aged sharp cheddar", "quail", "filet mignon",
         "orange roughy", "tuna", "mussels", "oysters", "figs",
         "roasted tomatoes", "nutmeg", "curry", "thyme", "lemon butter",
         "alfredo sauce", "milk chocolate", "vanilla pudding",
         "strawberries and cream",
         # H-E-B additions:
         "poultry", "pasta", "soft rich cheese"],
        ["earthy", "red fruit", "cherry", "raspberry", "silky", "light",
         "elegant", "medium-bodied", "strawberry", "mushroom", "forest floor",
         "dusty", "oregon"],
    ),
    "gamay": (   # primary grape of Beaujolais
        ["pork", "roast chicken", "duck", "charcuterie", "salami", "ham",
         "thanksgiving", "salad", "antipasto", "tomato pasta", "italian",
         "pizza"],
        ["fruity", "light", "low tannin", "bright", "raspberry", "cherry",
         "floral", "beaujolais"],
    ),

    # =========================================================================
    # LOW-TANNIN REDS for TOMATO / SPICY FOOD
    # =========================================================================
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
         "tomato pasta", "ragu", "salumi",
         # H-E-B (Chianti / Sangiovese): Beef, Pasta, Tomatoes
         "beef", "tomatoes"],
        ["bright", "cherry", "herbal", "acidic", "tomato-friendly",
         "red fruit", "medium-bodied"],
    ),
    "montepulciano": (
        ["pizza", "pasta", "tomato sauce", "italian", "ragu", "lasagna",
         "spaghetti", "meatballs", "marinara"],
        ["cherry", "red berries", "low tannin", "fruity", "medium-bodied",
         "approachable"],
    ),
    "barbera": (
        ["pizza", "pasta", "tomato sauce", "italian", "salami", "pancetta",
         "polenta", "charcuterie", "antipasti", "beef stew"],
        ["bright", "cherry", "high acid", "low tannin", "italian",
         "piedmont"],
    ),

    # =========================================================================
    # MEDIUM / STRUCTURED REDS
    # =========================================================================
    "merlot": (
        ["roast chicken", "pork", "beef stew", "pasta", "mushroom", "duck",
         "young cheddar", "gouda", "mild cheese", "grilled tuna", "tuna steaks",
         "miso salmon", "charcuterie", "salami", "soup", "ham", "spiral ham",
         "honey ham",
         # Thermador additions:
         "chestnuts", "romano", "herb cheese", "grilled meats", "leg of lamb",
         "pancetta", "grilled swordfish", "tuna", "shrimp",
         "caramelized onions", "endive", "black beans", "asparagus", "mint",
         "rosemary", "thyme", "nutmeg", "bolognese", "bearnaise", "berries",
         "chocolate torte", "chocolate mousse",
         # H-E-B: simple beef
         "simple beef"],
        ["soft", "plum", "chocolate", "round", "medium-bodied", "smooth",
         "blueberry", "approachable", "vanilla", "oak"],
    ),
    "tempranillo": (
        ["lamb", "tapas", "chorizo", "paella", "hard cheese", "grilled meats",
         "spanish", "manchego", "rioja",
         # H-E-B (Rioja): Poultry, Tomatoes pasta
         "poultry", "tomato pasta", "roast pork"],
        ["leather", "tobacco", "cherry", "earthy", "spanish", "structured"],
    ),
    "red blend": (
        ["steak", "pasta", "burgers", "roast", "grilled meats", "lamb",
         "swordfish", "veal chops", "bbq pork", "pork belly",
         "roasted vegetables", "coq au vin", "chicken mole", "teriyaki",
         "chimichurri", "braised short ribs", "bbq ribs", "carne asada",
         "grilled sausage"],
        ["versatile", "balanced", "medium-bodied", "fruit-forward",
         "blackberry", "lavender", "rosemary", "silky tannin", "complex"],
    ),
    "bordeaux-style red blend": (
        ["steak", "lamb", "roast", "hard cheese", "duck", "lamb kebabs",
         "blackened swordfish", "veal", "bbq pork buns", "venison",
         # H-E-B (Red Bordeaux): Beef, Lamb
         "beef", "roast beef",
         # Thermador (Meritage): Tri-Tip, Osso Buco, Roast Beef, Tuna,
         # Grilled Shark, Crab, Lobster, Boysenberry, Bell Peppers, Cauliflower,
         # Black Pepper, Thyme, Lavender, Garlic, Mushroom, Bearnaise, Marinara,
         # Berry Cobbler, Chocolate Truffle
         "tri-tip", "osso buco", "grilled shark", "crab", "lobster",
         "boysenberry", "bell peppers", "cauliflower", "lavender",
         "mushroom sauce", "marinara", "berry cobbler", "chocolate truffle",
         "toasted cashews", "roquefort"],
        ["structured", "tannic", "blackcurrant", "oak", "elegant", "plum",
         "cedar", "vanilla", "dark fruit"],
    ),
    "meritage": (   # American Bordeaux-style blend
        ["tri-tip", "osso buco", "roast beef", "tuna", "grilled shark",
         "crab", "lobster", "boysenberry", "bell peppers", "cauliflower",
         "black pepper", "thyme", "lavender", "garlic", "mushroom",
         "bearnaise", "marinara", "berry cobbler", "chocolate truffle",
         "toasted cashews", "roquefort", "steak", "lamb"],
        ["structured", "tannic", "complex", "bordeaux-style", "balanced",
         "oak", "dark fruit"],
    ),
    "portuguese red": (
        ["grilled meats", "stew", "lamb", "sausage"],
        ["earthy", "dark fruit", "rustic", "structured"],
    ),

    # =========================================================================
    # BOLD / TANNIC REDS for FATTY MEAT
    # =========================================================================
    "cabernet sauvignon": (
        ["steak", "ribeye", "lamb", "burger", "hard cheese", "aged cheddar",
         "bbq", "tomahawk", "porterhouse", "dry-aged ribeye", "strip steak",
         "crown pork", "london broil", "prime rib", "large roasts",
         "hearty stews", "beef stew", "venison stew", "grilled vegetables",
         "dark chocolate", "chocolate truffles", "blue cheese",
         # Thermador additions:
         "pralines", "hazelnuts", "gorgonzola", "sharp cheddar", "venison",
         "duck", "veal", "rare tuna", "swordfish", "cioppino",
         "black cherries", "squash", "plums", "broccoli", "rosemary",
         "tarragon", "marjoram", "meat stock", "sesame", "olive oil",
         "red pepper cream", "bittersweet chocolate", "espresso", "gelato",
         # H-E-B: Beef, Lamb, Hard Flavorful cheese
         "beef", "hard flavorful cheese"],
        ["bold", "tannic", "blackcurrant", "cassis", "structured", "powerful",
         "oak", "full-bodied", "rich tannin", "dark fruit", "complex",
         "herbaceous", "spice"],
    ),
    "syrah": (
        ["bbq", "ribs", "lamb", "venison", "grilled meats", "stew",
         "steak", "shiraz steak", "braised meats", "moroccan", "duck",
         # Thermador additions:
         "hazelnuts", "roquefort", "spicy sausage", "pork chops",
         "game hens", "halibut", "salmon", "red snapper", "stewed tomatoes",
         "beets", "currants", "cherries", "sage", "thyme", "garlic",
         "parsley", "red sauce", "spicy", "steak sauce", "vodka sauce",
         "black forest cake", "chocolate", "caramel",
         # H-E-B: Beef
         "beef"],
        ["smoky", "peppery", "spicy", "dark fruit", "full-bodied",
         "blackberry", "leather"],
    ),
    "shiraz": (
        ["bbq", "ribs", "grilled meats", "stew", "burgers", "steak",
         "lamb roast", "beef bourguignon", "dark chocolate", "hard cheese",
         "dried fruit",
         # shared with Syrah:
         "spicy sausage", "pork chops", "halibut", "salmon", "red snapper",
         "stewed tomatoes", "beets", "cherries", "sage", "thyme", "garlic",
         "red sauce", "steak sauce", "vodka sauce", "black forest cake",
         "beef"],
        ["jammy", "peppery", "bold", "dark fruit", "ripe", "full-bodied",
         "smoky", "blackberry"],
    ),
    "zinfandel": (
        ["bbq", "pizza", "burgers", "pulled pork", "spiced meats", "ribs",
         # Thermador additions:
         "muenster", "manchego", "spicy sausage", "beef", "lamb",
         "cioppino", "mahi mahi", "cranberries", "grilled peppers",
         "olives", "black pepper", "thyme", "bay leaf", "garlic", "spicy",
         "cajun", "hot salsa", "chimichurri", "spice cake", "carrot cake",
         "bread pudding", "panna cotta",
         # H-E-B: Beef, Tomato Sauce pasta
         "tomato sauce", "tomato pasta"],
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
         "porcini", "wild boar",
         # H-E-B (Barolo / Nebbiolo): Beef, Lamb, Lasagne
         "beef", "lasagne", "pork"],
        ["tannic", "rose", "tar", "cherry", "earthy", "long-aging",
         "high tannin", "structured", "piedmont"],
    ),

    # =========================================================================
    # ROSÉ / SPARKLING / DESSERT
    # =========================================================================
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
         "salads", "birthday cake", "fruit tarts", "chocolate-covered fruit",
         # Thermador additions:
         "parmesan", "gouda", "almonds", "pecans", "spicy meats", "crab",
         "mushrooms", "berries", "butternut squash", "cinnamon", "sage",
         "mustard", "cream sauce", "smokey bbq", "olive oil", "shortbread",
         "creme brulee", "lemon tart"],
        ["sparkling", "crisp", "bright", "yeasty", "citrus", "celebration",
         "lemon", "orange", "apple", "pear", "stone fruit", "clean"],
    ),
    "sparkling blend": (
        ["oysters", "fried food", "popcorn", "brunch", "sushi", "appetizers",
         "shellfish", "prawns", "thai prawns", "tapas", "paella",
         "artichokes", "asparagus", "salami", "italian hard cheese",
         "asiago", "provolone", "charcuterie", "crusty bread", "caviar"],
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
         "sorbet", "strawberries",
         # Thermador additions:
         "macadamia nuts", "mascarpone", "charcuterie", "turkey", "duck",
         "crab", "trout", "scallops", "cod", "chili pepper", "fig", "mango",
         "pineapple", "ginger", "black pepper", "clove", "cayenne pepper",
         "spicy", "sweet and sour", "brown butter", "sage sauce",
         "biscotti", "macadamia cookies", "chocolate"],
        ["sweet", "light-bodied", "apricot", "nectarine", "grapefruit",
         "honeydew", "mandarin", "off-dry", "aromatic"],
    ),
    "port": (
        ["chocolate", "blue cheese", "nuts", "dessert", "dark chocolate",
         "stilton", "walnuts", "fig", "cigar"],
        ["sweet", "rich", "fortified", "dark fruit", "dessert"],
    ),
    "sauternes": (   # sweet Bordeaux dessert wine
        ["blue cheese", "roquefort", "stilton", "foie gras", "fruit tart",
         "creme brulee", "lemon tart", "apricot tart", "spicy thai",
         "spicy indian"],
        ["sweet", "honeyed", "botrytis", "apricot", "honey", "dessert wine",
         "rich"],
    ),
}


# =============================================================================
# 2. Cuisine -> list of well-pairing varieties
# =============================================================================
CUISINE_VARIETIES: Dict[str, List[str]] = {
    "italian": [
        "sangiovese", "montepulciano", "nebbiolo", "barbera", "pinot grigio",
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
        "sauvignon blanc", "chardonnay",
    ],
    "chinese": [
        "riesling", "pinot noir", "gewürztraminer",
    ],
    "mexican": [
        "grenache", "zinfandel", "pinot noir", "riesling",
    ],
    "french": [
        "pinot noir", "chardonnay", "bordeaux-style red blend",
        "sauvignon blanc", "champagne blend", "gamay", "chablis",
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
    "asian/indian": [
        "riesling", "gewürztraminer", "sauvignon blanc", "chardonnay",
        "pinot grigio",
    ],
}


# =============================================================================
# 3. Pairing principles (sauce / dish-element -> wine attribute)
# =============================================================================
# Captures the rules-of-thumb from the Wine Matchmaker guide and the Court of
# Master Sommeliers Europe principles doc, so the recommender can reason about
# *why* certain matches work, not just memorized lists.
PAIRING_PRINCIPLES: Dict[str, str] = {
    # ---- fat / acid / tannin axis ----
    "fatty":         "tannin or high acid to cut through the fat",
    "salty":         "off-dry wine or dry sherry; high-tannin reds will clash, "
                     "but salt pairs beautifully with sweet or fruity wines",
    "spicy":         "off-dry whites (Riesling, Gewürztraminer) or low-tannin "
                     "fruity reds (Grenache); high-tannin reds clash with chili, "
                     "and high-alcohol wines amplify the heat",
    "creamy":        "rich whites with body and oak (Chardonnay) or off-dry; "
                     "acidity in the wine contrasts and refreshes the cream",
    "tomato":        "low-tannin reds (Grenache, Sangiovese, Montepulciano, "
                     "Barbera); high-tannin reds turn metallic against tomato acid",
    "smoky":         "Syrah, Shiraz, Malbec, smoky reds",
    "sweet dessert": "wine must be sweeter than the dessert (Moscato, Port, "
                     "Sauternes, late-harvest, botrytis)",
    "raw seafood":   "high-acid crisp whites or sparkling (Riesling, Sauvignon "
                     "Blanc, Champagne, Chablis); avoid tannic reds — fish + "
                     "tannin = metallic aftertaste",
    "grilled meat":  "tannic full-bodied reds (Cabernet, Syrah, Shiraz, "
                     "Nebbiolo, Malbec, Meritage); protein and fat soften tannins",
    "soft cheese":   "Sauvignon Blanc, Champagne, Pinot Gris, Chardonnay",
    "hard cheese":   "tannic reds (Cabernet, Nebbiolo, aged Pinot Noir, "
                     "Meritage)",
    "blue cheese":   "Sauternes, Port, or sweet wines — the sweetness balances "
                     "the cheese's funk and salt",
    # ---- structure principles from CMS Europe ----
    "match weight":  "match the body of the wine to the body of the food — "
                     "light dish, light wine; rich dish, full-bodied wine",
    "acid match":    "wine's acidity must match or exceed the food's acidity, "
                     "or the wine will taste flabby",
    "sugar match":   "wine's sweetness must match or exceed the food's sweetness, "
                     "or the wine will taste bitter",
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
    if any(w in s for w in ("steak", "ribeye", "grill", "bbq", "lamb",
                            "porterhouse", "tomahawk", "prime rib")):
        return PAIRING_PRINCIPLES["grilled meat"]
    if any(w in s for w in ("oyster", "sushi", "ceviche", "sashimi", "raw")):
        return PAIRING_PRINCIPLES["raw seafood"]
    if any(w in s for w in ("curry", "chili", "thai", "indian", "szechuan",
                            "spicy", "cajun", "harissa")):
        return PAIRING_PRINCIPLES["spicy"]
    if any(w in s for w in ("brie", "camembert", "goat cheese", "burrata",
                            "mozzarella")):
        return PAIRING_PRINCIPLES["soft cheese"]
    if any(w in s for w in ("cheddar", "parmesan", "aged", "manchego",
                            "gouda", "asiago", "gruyere")):
        return PAIRING_PRINCIPLES["hard cheese"]
    if any(w in s for w in ("blue cheese", "roquefort", "stilton",
                            "gorgonzola")):
        return PAIRING_PRINCIPLES["blue cheese"]
    if any(w in s for w in ("dessert", "cake", "tart", "chocolate",
                            "pie", "ice cream")):
        return PAIRING_PRINCIPLES["sweet dessert"]
    if any(w in s for w in ("cream sauce", "alfredo", "carbonara",
                            "bechamel", "butter sauce")):
        return PAIRING_PRINCIPLES["creamy"]
    if any(w in s for w in ("tomato", "marinara", "bolognese")):
        return PAIRING_PRINCIPLES["tomato"]
    return None
