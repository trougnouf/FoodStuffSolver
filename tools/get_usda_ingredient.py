"""
This tool gets an ingredient from the USDA database and saves it as a yaml file
under ingredients.

The nutrients match those defined in libnutrient.py and the format matches that
used in libingredient.py.
"""
from typing import Union

import requests
import os
import sys

sys.path.append(".")
import libnutrient
import libingredient

API_KEY_FPATH = "usda_api.key"

# TODO add comment to recipe, include missing nutrients to comment

NUTRIENTS_USDA_NAMES = {
    "Energy": ["Energy (Atwater Specific Factors)", "Energy"],
    "Carbohydrate": ["Carbohydrate, by difference"],
    "Omega-3 fatty acid": ["n-3", "PUFA 18:3"],
    "Omega-6 fatty acid": ["n-6", "PUFA 18:2"],
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
    ignored_nutrients = []
    preprocess_ingredient_dict(ingredient_dict)
    for usda_nutrient_name, usda_nutrient_dict in ingredient_dict["nutrients"].items():
        qty = usda_nutrient_dict["qty"]
        if qty == 0:
            continue
        try:
            nutrient = get_nutrient_obj(usda_nutrient_name)
        except ValueError:
            print(f"Warning: nutrient {usda_nutrient_name} not found in libnutrient.py")
            ignored_nutrients.append(usda_nutrient_name)
            continue
        if nutrient.unit != usda_nutrient_dict["unit"]:
            qty = nutrient.convert_qty(qty, usda_nutrient_dict["unit"])
        qty /= 100  # convert to g
        if nutrient not in nutrients_qty:
            nutrients_qty[nutrient] = qty
        else:
            print(
                f'Warning: adding nutrient "{nutrient.name}" twice. Existing qty: {nutrients_qty[nutrient]}. Additional qty: {qty}'
            )
            nutrients_qty[nutrient] += qty
    ingredient = libingredient.Ingredient(
        name=ingredient_dict["name"],
        nutrients_qty=nutrients_qty,
        food_id=ingredient_dict["food_id"],
        comment=f"Imported from USDA database. Ignored nutrients: {'; '.join(ignored_nutrients)}",
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
        # Test with Sunflower Seeds (SR Legacy) which has both 18:2 and 18:3
        food_id = 170562
    ingredient_id_to_yaml(food_id)
