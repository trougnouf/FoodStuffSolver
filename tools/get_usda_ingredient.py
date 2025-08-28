"""
This tool gets an ingredient from the USDA database and saves it as a yaml file
under ingredients.

The nutrients match those defined in libnutrient.py and the format matches that
used in libingredient.py.
"""
from typing import Union
from collections import defaultdict
import re
import os
import sys
import requests

sys.path.append(".")
import libnutrient
import libingredient

API_KEY_FPATH = "usda_api.key"

NUTRIENTS_USDA_NAMES = {
    "Energy": ["Energy (Atwater Specific Factors)", "Energy"],
    "Carbohydrate": ["Carbohydrate, by difference"],
    "Omega-3 fatty acid": ["n-3", "PUFA 18:3"],
    "Omega-6 fatty acid": ["n-6", "PUFA 18:2", "PUFA 20:4"],
    "Vitamin A, RAE": ["Vitamin A, IU"],
    "Vitamin C": ["Vitamin C, total ascorbic acid"],
    "Vitamin D": ["Vitamin D (D2 + D3), International Units"],
    "Vitamin E": ["Vitamin E (alpha-tocopherol)"],
    "Vitamin K": ["Vitamin K (phylloquinone)"],
    "Folate": ["Folate, total"],
    "Choline": ["Choline, total"],
}


def get_nutrient_obj(name: str) -> libnutrient.Nutrient:
    """Returns the nutrient object corresponding to the given nutrient name."""
    if name in libnutrient.NUTRIENTS:
        return libnutrient.NUTRIENTS[name]
    for nutrient_std_name, nutrient_alt_names in NUTRIENTS_USDA_NAMES.items():
        if any(usda_name in name for usda_name in nutrient_alt_names):
            return libnutrient.NUTRIENTS[nutrient_std_name]
    raise ValueError(f"Nutrient {name} not found in libnutrient.py")


def read_usda_api_key(api_key_fpath: str = API_KEY_FPATH) -> str:
    """Return the USDA API key saved on disk."""
    if not os.path.isfile(api_key_fpath):
        raise FileNotFoundError(
            (
                "Missing USDA API key; get one from "
                "https://fdc.nal.usda.gov/api-key-signup.html and save it in "
                f"{api_key_fpath}"
            )
        )
    with open(api_key_fpath, "r") as fp:
        api_key = fp.read()
        print(f"Using api_key ending in {api_key[-4:]}")
    return api_key


def get_usda_ingredient_nutrients(
    food_id: str,
) -> dict[dict[str, str], dict[str, dict[str, Union[float, str]]]]:
    """
    Gets an ingredient from the USDA database using the specified food_id and
    returns a dictionary of the ingredient's name and nutrients with their
    quantity and unit."""
    api_key = read_usda_api_key()
    url = f"https://api.nal.usda.gov/fdc/v1/food/{food_id}?api_key={api_key}"
    res = requests.get(url).json()
    if "errors" in res:
        raise ValueError(
            f"Error getting ingredient from USDA database: {res['errors']}"
        )
    ingredient = res
    nutrients = {
        nutrient["nutrient"]["name"]: {
            "qty": nutrient["amount"],
            "unit": nutrient["nutrient"]["unitName"],
        }
        for nutrient in ingredient["foodNutrients"]
        if "amount" in nutrient
    }
    return {
        "name": ingredient["description"],
        "nutrients": nutrients,
        "food_id": food_id,
    }


def preprocess_ingredient_dict(
    ingredient_dict: dict[dict[str, str], dict[str, dict[str, Union[float, str]]]]
) -> None:
    """
    Preprocesses the ingredient dictionary to make it compatible with libingredient.py.
    """
    if (
        "Energy (Atwater General Factors)" in ingredient_dict["nutrients"]
        and "Energy (Atwater Specific Factors)" in ingredient_dict["nutrients"]
    ):
        del ingredient_dict["nutrients"]["Energy (Atwater General Factors)"]
    if ("Vitamin A, IU" in ingredient_dict["nutrients"]) and (
        "Vitamin A, RAE" in ingredient_dict["nutrients"]
    ):
        del ingredient_dict["nutrients"]["Vitamin A, IU"]


def ingredient_dict_to_obj(
    ingredient_dict: dict[dict[str, str], dict[str, dict[str, Union[float, str]]]]
) -> libingredient.Ingredient:
    """
    Converts a dictionary of an ingredient's name and nutrients with their
    quantity and unit to an Ingredient object.
    """
    nutrients_qty = {}
    unmapped_nutrients = []
    ignored_as_less_specific = []
    
    preprocess_ingredient_dict(ingredient_dict)

    # Step 1: Group all raw USDA nutrients by their standard Nutrient object.
    grouped_nutrients = defaultdict(list)
    for usda_nutrient_name, usda_nutrient_dict in ingredient_dict["nutrients"].items():
        if usda_nutrient_dict["qty"] == 0:
            continue
        try:
            nutrient_obj = get_nutrient_obj(usda_nutrient_name)
            grouped_nutrients[nutrient_obj].append({
                "name": usda_nutrient_name,
                "qty": usda_nutrient_dict["qty"],
                "unit": usda_nutrient_dict["unit"]
            })
        except ValueError:
            print(f"Warning: nutrient {usda_nutrient_name} not found in libnutrient.py")
            unmapped_nutrients.append(usda_nutrient_name)

    # Step 2: Process each group with the new prioritization logic.
    for nutrient_obj, nutrient_list in grouped_nutrients.items():
        final_nutrients_to_sum = []
        
        # Use advanced logic only for complex fatty acid groups
        if nutrient_obj.name in ["Omega-3 fatty acid", "Omega-6 fatty acid"]:
            # Sub-group nutrients by their base family (e.g., '18:2', '18:3')
            fatty_acid_families = defaultdict(list)
            for nutrient in nutrient_list:
                match = re.search(r'(\d+:\d+)', nutrient['name'])
                key = match.group(1) if match else nutrient['name']
                fatty_acid_families[key].append(nutrient)

            # For each family, choose the most specific entry (longest name)
            for family, members in fatty_acid_families.items():
                if not members: continue
                
                # The 'best' nutrient is assumed to be the one with the most specific name
                best_nutrient = max(members, key=lambda x: len(x['name']))
                final_nutrients_to_sum.append(best_nutrient)
                print(f"For family '{family}', chose '{best_nutrient['name']}' as most specific.")

                # Log the less specific ones that were ignored
                for member in members:
                    if member['name'] != best_nutrient['name']:
                        ignored_as_less_specific.append(member['name'])
        else:
            # For simple nutrients, just use the whole list
            final_nutrients_to_sum = nutrient_list

        # Step 3: Sum the quantities of the de-duplicated nutrients.
        total_qty = 0
        for nutrient_data in final_nutrients_to_sum:
            qty = nutrient_data["qty"]
            if nutrient_obj.unit != nutrient_data["unit"]:
                qty = nutrient_obj.convert_qty(qty, nutrient_data["unit"])
            total_qty += qty
        
        nutrients_qty[nutrient_obj] = total_qty / 100

    # Construct a more descriptive final comment
    comment_parts = ["Imported from USDA database."]
    if unmapped_nutrients:
        comment_parts.append(f"Unmapped nutrients: {'; '.join(sorted(unmapped_nutrients))}")
    if ignored_as_less_specific:
        comment_parts.append(f"Ignored less-specific entries: {'; '.join(sorted(ignored_as_less_specific))}")
    
    final_comment = " ".join(comment_parts)

    ingredient = libingredient.Ingredient(
        name=ingredient_dict["name"],
        nutrients_qty=nutrients_qty,
        food_id=ingredient_dict["food_id"],
        comment=final_comment,
    )
    return ingredient


def ingredient_id_to_yaml(ingredient_id: str) -> None:
    """
    Gets an ingredient from the USDA database using the specified food_id and
    saves it as a yaml file under ingredients.
    """
    ingredient_dict = get_usda_ingredient_nutrients(ingredient_id)
    ingredient_obj = ingredient_dict_to_obj(ingredient_dict)
    ingredient_obj.save_to_yaml()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        food_id = sys.argv[1]
    else:
        # Test with Peanuts (Branded) which has the complex duplicate issue
        food_id = 173806
    ingredient_id_to_yaml(food_id)
