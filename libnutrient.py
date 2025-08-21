import yaml
from typing import Optional, Dict

class Nutrient:
    def __init__(
        self,
        name: str,
        unit: str,
        rdi: float,
        ul: float,
    ):
        self.name = name
        self.unit = unit
        self.rdi = rdi
        self.ul = ul
        self._conversion_functions = {
            ("g", "mg"): lambda qty: qty * 1000,
            ("mg", "µg"): lambda qty: qty * 1000,
            ("mg", "g"): lambda qty: qty / 1000,
            ("µg", "mg"): lambda qty: qty / 1000,
            ("µg", "ug"): lambda qty: qty,
            ("ug", "µg"): lambda qty: qty,
        }

    def convert_qty(self, qty: float, unit_from: str):
        try:
            return self._conversion_functions[(unit_from, self.unit)](qty)
        except KeyError:
            raise NotImplementedError(
                'Conversion from "{}" to "{}" not implemented for {}'.format(
                    unit_from, self.unit, self.name
                )
            )

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'Nutrient(name="{self.name}", unit="{self.unit}", rdi={self.rdi}, ul={self.ul})'

    def __hash__(self):
        return hash(self.name)

# This global variable will hold the loaded nutrient profile.
NUTRIENTS: Dict[str, Nutrient] = {}

def get_default_nutrient_definitions():
    """Returns the hardcoded nutrient definitions with original comments."""
    # Each item is a tuple: (Name, Unit, RDI, UL, Comment)
    return [
        ("Energy", "kcal", 700, 2150, "Calorie goals per meal (assuming 3 meals/day)"),
        ("Carbohydrate", "g", 0, None, ""),
        ("Protein", "g", 112, None, "Based on 1.6 g/kg for a 70 kg person"),
        ("Total lipid (fat)", "g", 0, None, ""),
        ("Fatty acids, total saturated", "g", 0, None, ""),
        ("Fatty acids, total monounsaturated", "g", 0, None, ""),
        ("Fatty acids, total polyunsaturated", "g", 0, None, ""),
        ("Omega-3 fatty acid", "g", 3, None, "Source: Dose-Response Meta-Analysis of RCTs"),
        ("Omega-6 fatty acid", "g", 0, None, ""),
        ("Fiber, total dietary", "g", 28, None, ""),
        ("Cholesterol", "mg", 0, None, ""),
        ("Vitamin A, RAE", "µg", 1350, 3000, "NAM (2024), adjusted for 2/3 daily intake"),
        ("Vitamin B-6", "mg", 1.7, 100, "NAM, EFSA (2024)"),
        ("Vitamin B-12", "µg", 4.0, None, "EFSA (2024)"),
        ("Vitamin C", "mg", 330, 2000, "EFSA (2024) is 110mg; tripled for per-meal RDI"),
        ("Vitamin D", "IU", 900, 4000, "Based on 600 IU govt rec, adjusted for 2/3 daily intake"),
        ("Vitamin E", "mg", 22.5, 300, "NAM (2024) is 15mg, adjusted for 2/3 daily intake"),
        ("Vitamin K", "µg", 180, None, "NAM (2024) is 120mg, adjusted for 2/3 daily intake"),
        ("Thiamin", "mg", 1.8, None, "NAM (2024) is 1.2mg, adjusted for 2/3 daily intake"),
        ("Riboflavin", "mg", 1.6, None, "EFSA (2024)"),
        ("Niacin", "mg", 24, 35, "NAM (2024) is 16mg, adjusted for 2/3 daily intake"),
        ("Folate", "µg", 600, 1000, "NAM (2024) is 400µg, adjusted for 2/3 daily intake"),
        ("Pantothenic acid", "mg", 7.5, None, "NAM, EFSA (2024) is 5mg, adjusted for 2/3 daily intake"),
        ("Biotin", "µg", 60, None, "EFSA (2024) is 40µg, adjusted for 2/3 daily intake"),
        ("Choline", "mg", 825, 3500, "NAM (2024) is 550mg, adjusted for 2/3 daily intake"),
        ("Calcium, Ca", "mg", 1500, 2500, "NAM (2024) is 1000mg, adjusted for 2/3 daily intake"),
        ("Chloride", "g", 0, 3.6, "Not tracked by USDA"),
        ("Chromium", "µg", 0, 35, "Not tracked by USDA"),
        ("Copper, Cu", "mg", 2.4, 5, "EFSA (2024) is 1.6mg, adjusted for 2/3 daily intake"),
        ("Iodine, I", "µg", 225, 1100, "NAM, EFSA (2024) is 150µg, adjusted for 2/3 daily intake"),
        ("Iron, Fe", "mg", 16.5, 45, "EFSA (2024) is 11mg, adjusted for 2/3 daily intake"),
        ("Magnesium, Mg", "mg", 630, None, "NAM (2024) is 420mg, adjusted for 2/3 daily intake"),
        ("Manganese, Mn", "mg", 4.5, 11, "EFSA (2024) is 3mg, adjusted for 2/3 daily intake"),
        ("Molybdenum, Mo", "µg", 0, 2000, "Barely tracked by USDA"),
        ("Phosphorus, P", "mg", 1050, 4000, "US Inst. of Med. is 700mg, adjusted for 2/3 daily intake"),
        ("Potassium, K", "mg", 5250, None, "EFSA (2024) is 3500mg, adjusted for 2/3 daily intake"),
        ("Selenium, Se", "µg", 105, 400, "EFSA (2024) is 70µg, adjusted for 2/3 daily intake"),
        ("Sodium, Na", "mg", 1750, 2300, "Between NAM (1500mg) and EFSA (2000mg)"),
        ("Sulfur", "g", 0, None, "Barely tracked by USDA"),
        ("Zinc, Zn", "mg", 16.3, 25, "NAM: 11, EFSA: 9.4-16.3 depending on phytate intake"),
    ]

def _apply_special_conversions(nutrients_dict: Dict[str, Nutrient]):
    """Applies known special unit conversions to a dictionary of Nutrient objects."""
    if "Vitamin A, RAE" in nutrients_dict:
        nutrients_dict["Vitamin A, RAE"]._conversion_functions[("IU", "µg")] = lambda qty: qty * 0.3
    if "Vitamin E" in nutrients_dict:
        nutrients_dict["Vitamin E"]._conversion_functions[("IU", "mg")] = lambda qty: qty * 0.67
    if "Energy" in nutrients_dict:
        nutrients_dict["Energy"]._conversion_functions[("kJ", "kcal")] = lambda qty: qty / 4.184
    return nutrients_dict

def initialize_nutrients(fpath: Optional[str] = None):
    """
    Loads nutrient definitions from a YAML file or uses defaults,
    then populates the global NUTRIENTS dictionary.
    """
    global NUTRIENTS
    
    nutrient_data = {}
    if fpath:
        print(f"Loading custom nutrient profile from: {fpath}")
        with open(fpath, 'r', encoding="utf-8") as f:
            nutrient_data = yaml.safe_load(f)
    else:
        print("Loading default nutrient profile.")
        # Convert default list of tuples to the same dict format as the YAML
        for name, unit, rdi, ul, _ in get_default_nutrient_definitions():
            nutrient_data[name] = {'unit': unit, 'rdi': rdi, 'ul': ul}

    # Create Nutrient objects from the loaded data
    temp_nutrients = {}
    for name, properties in nutrient_data.items():
        temp_nutrients[name] = Nutrient(
            name=name,
            unit=properties['unit'],
            rdi=properties['rdi'],
            ul=properties.get('ul') # Use .get for optional 'ul'
        )
    
    # Apply special conversions and update the global dict
    NUTRIENTS = _apply_special_conversions(temp_nutrients)

# Initialize with defaults when the module is first imported.
# This can be overridden later by calling initialize_nutrients() again.
initialize_nutrients()


if __name__ == "__main__":
    # This block allows you to generate a template YAML file.
    # Run: python libnutrient.py > default_nutrients.yaml
    

    definitions = get_default_nutrient_definitions()
    
    yaml_dict = {}
    for name, unit, rdi, ul, _ in definitions:
        entry = {'unit': unit, 'rdi': rdi}
        # Only include 'ul' if it's not None for a cleaner file
        if ul is not None:
            entry['ul'] = ul
        yaml_dict[name] = entry

    print("# Default Nutrient Profile for Recipe Solver")
    print("# Generated by `python libnutrient.py`\n")
    
    print(yaml.dump(
        yaml_dict, 
        sort_keys=False, 
        indent=2, 
        allow_unicode=True # This correctly handles characters like 'µ'
    ))
