import os
import yaml
import unittest
import math
from typing import Any, Optional, Union
import libnutrient

INGREDIENTS_DIR = "ingredients"
# MIN_COST_STEP = 0.001
# TODO add fixed min. cost based on expiration


class Ingredient:
    def __init__(
        self,
        name: str,
        nutrients_qty: dict[libnutrient.Nutrient, float],
        cost_per_unit: Optional[
            float
        ] = None,  # TODO warn user if using no-cost ingredient
        unit: str = "g",
        step_qty: float = 1,
        source: Optional[str] = None,
        max_qty: Optional[float] = None,
        fpath: Optional[str] = None,
        food_id: Optional[Union[str, int]] = None,
        comment: Optional[str] = None,
        allergen: list[str] = [],
        # min_cost_step: Optional[float] = MIN_COST_STEP,
        satisfaction_multiplier: int = 1,
    ):
        self.name = name
        self.cost_per_unit = cost_per_unit  # really cost per gram
        self.unit = unit
        self.step_qty = step_qty
        self.source = source or ""
        self.max_qty = max_qty
        self.nutrients_qty: dict[libnutrient.Nutrient, float] = nutrients_qty
        self.fpath = fpath or os.path.join(INGREDIENTS_DIR, f"{name}.yaml")
        self.food_id = food_id
        self.comment = comment
        self.allergen = allergen
        # self.step_qty = 1  # dbg
        # print("warning: hardcoded step_qty=1 in Ingredient.__init__")
        # self.min_cost_step = min_cost_step
        self.satisfaction_multiplier = satisfaction_multiplier

    def save_to_yaml(self, fpath: Optional[str] = None):
        fpath = fpath or self.fpath
        if not fpath:
            raise IOError("No file path provided")
        if os.path.isfile(fpath):
            raise FileExistsError(f"File already exists: {fpath}")
        ingredient_data = {
            "name": self.name,
            "cost_per_unit": self.cost_per_unit,
            "unit": self.unit,
            "step_qty": self.step_qty,
            "source": self.source,
            "max_qty": self.max_qty,
            "nutrients_qty": {
                nutrient.name: nutrient_qty
                for nutrient, nutrient_qty in self.nutrients_qty.items()
            },
            "food_id": self.food_id,
            "comment": self.comment,
            "allergen": self.allergen,
            # "min_cost_step": self.min_cost_step,
            "satisfaction_multiplier": self.satisfaction_multiplier,
        }
        with open(fpath, "w") as f:
            yaml.safe_dump(ingredient_data, f)
            print(f"Saved ingredient to {fpath}")

    # def is_significant(self, qty: float):

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def load_from_yaml(cls, fpath):
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find file: {fpath}")
        with open(fpath, "r") as f:
            fdata = yaml.safe_load(f)
        return cls(
            name=fdata["name"],
            cost_per_unit=fdata["cost_per_unit"],
            unit=fdata["unit"],
            step_qty=fdata["step_qty"],
            nutrients_qty={
                libnutrient.NUTRIENTS[nutrient_name]: nutrient_qty
                for nutrient_name, nutrient_qty in fdata.pop(
                    "nutrients_qty", {}
                ).items()
            },
            max_qty=fdata.pop("max_qty", None),
            source=fdata.pop("source", ""),
            fpath=fpath,
            food_id=fdata.pop("food_id", None),
            comment=fdata.pop("comment", None),
            allergen=fdata.pop("allergen", []),
            # min_cost_step=fdata.pop("min_cost_step", MIN_COST_STEP),
            satisfaction_multiplier=fdata.pop("satisfaction_multiplier", 1),
        )

    def calculate_cost(self, qty: float, true_cost: bool = False):
        if true_cost:
            satisfaction_multiplier = 1
        else:
            satisfaction_multiplier = self.satisfaction_multiplier
        return (qty * self.cost_per_unit) / satisfaction_multiplier
        # if qty < 0.0001:  # TODO replace w/ significance check
        #     return (qty * self.cost_per_unit) / satisfaction_multiplier
        # return (
        #     (math.ceil(qty / self.min_cost_step) * self.min_cost_step)
        #     * self.cost_per_unit
        # ) / satisfaction_multiplier


def get_ingredients(
    dpaths: Union[list[str], str] = INGREDIENTS_DIR, allergies: list[str] = []
):
    """get all ingredients from ingredients directory"""
    ingredients = []
    if isinstance(dpaths, str):
        dpaths = [dpaths]
    for dpath in dpaths:
        for fname in os.listdir(dpath):
            if fname.endswith(".yaml"):
                ingredient = Ingredient.load_from_yaml(os.path.join(dpath, fname))
                if not any(allergy in ingredient.allergen for allergy in allergies):
                    ingredients.append(ingredient)
    return ingredients


def augment_ingredients(
    ingredients: list[Ingredient], ingredients_augmentations: dict[str, dict[str, Any]]
):
    for (
        ingredient_to_augment,
        ingredient_augmentations,
    ) in ingredients_augmentations.items():
        for ingredient in ingredients:
            if ingredient.name == ingredient_to_augment:
                for (
                    augmentation_name,
                    augmentation_value,
                ) in ingredient_augmentations.items():
                    if augmentation_name == "ingredients_qty":
                        for nutrient_name, nutrient_qty in augmentation_value.items():
                            nutrient = libnutrient.NUTRIENTS[nutrient_name]
                            ingredient.nutrients_qty[nutrient] = nutrient_qty
                    else:
                        setattr(ingredient, augmentation_name, augmentation_value)
                break
        else:
            # warn user that augmentation was not used and ask if they'd like to continue
            if (
                input(
                    f"Warning: {ingredient_to_augment} not found in ingredients. Continue? [Y/n]"
                )
                == "n"
            ):
                raise ValueError(
                    f"Ingredient {ingredient_to_augment} not found in ingredients"
                )


# not used
# def get_ingredient_by_name(ingredients: list[Ingredient], name: str):
#     for ingredient in ingredients:
#         if ingredient.name == name:
#             return ingredient
#     raise ValueError(f"Could not find ingredient with name: {name}")


class TestIngredient(unittest.TestCase):
    def test_load_from_save_to_yaml(self):
        """check that loading an ingredient and saving it produces the same yaml file"""
        ingredient = Ingredient.load_from_yaml(
            os.path.join(INGREDIENTS_DIR, "Flaxseed, ground.yaml")
        )
        ingredient.save_to_yaml("test.yaml")
        with open("test.yaml", "r") as f:
            fdata = yaml.safe_load(f)
            with open(os.path.join(INGREDIENTS_DIR, "Flaxseed, ground.yaml"), "r") as f:
                fdata2 = yaml.safe_load(f)
                self.assertEqual(fdata, fdata2)
        os.remove("test.yaml")


if __name__ == "__main__":
    unittest.main()
