import yaml
import sys
import statistics
import copy
import math
import libingredient
import libnutrient
import sys


MAX_INGREDIENT_CAL = (
    500  # avoid using an ingredient for more than 500 kcal per day TODO move
)
OMEGA_36_RATIO = 0.5


class Recipe:
    def __init__(
        self, ingredients_qty: dict[libingredient.Ingredient, float], n_steps=0
    ):
        self.ingredients_qty = ingredients_qty
        self.nutrients_qty: dict[libnutrient.Nutrient, float] = {
            nutrient: 0 for nutrient in libnutrient.NUTRIENTS.values()
        }
        self.cost: float = 0
        for ingredient, ingredient_qty in self.ingredients_qty.items():
            for nutrient, nutrient_qty in ingredient.nutrients_qty.items():
                self.nutrients_qty[nutrient] += ingredient_qty * nutrient_qty
            self.cost += ingredient.calculate_cost(ingredient_qty)
        self._compute_completeness_score()
        self.n_steps = n_steps

    def expand(self, ingredient: libingredient.Ingredient, qty: int) -> "Recipe":
        """Used by the naive solver"""
        new_recipe = copy.copy(self)
        new_recipe.ingredients_qty = self.ingredients_qty.copy()
        new_recipe.ingredients_qty[ingredient] += qty
        new_recipe.nutrients_qty = self.nutrients_qty.copy()
        for nutrient, nutrient_qty in ingredient.nutrients_qty.items():
            new_recipe.nutrients_qty[nutrient] += qty * nutrient_qty
        new_recipe.cost = self.cost + ingredient.calculate_cost(qty)
        new_recipe.n_steps = self.n_steps + 1
        new_recipe._compute_completeness_score()  # maybe this can be optimized?
        return new_recipe

    def _compute_completeness_score(self):
        completeness_scores = []
        for nutrient, qty in self.nutrients_qty.items():
            if nutrient.rdi:
                completeness_scores.append(min(qty, nutrient.rdi) / nutrient.rdi)
        omega3_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-3 fatty acid"]]
        omega6_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-6 fatty acid"]]
        if omega6_qty == 0:
            if omega3_qty == 0:
                completeness_scores.append(0)
            else:
                completeness_scores.append(1)
        else:
            completeness_scores.append(min(omega3_qty * 2 / omega6_qty, 1))
        self.completeness_score = statistics.mean(completeness_scores)

    def calculate_true_cost(self) -> float:
        """Return true cost w/o accounting for satisfaction multiplier"""
        true_cost = 0
        for ingredient, qty in self.ingredients_qty.items():
            true_cost += ingredient.calculate_cost(qty, true_cost=True)
        return true_cost

    def is_safe(self) -> bool:
        # no >UL and _check_flaxseed_amt
        for nutrient, qty in self.nutrients_qty.items():
            if nutrient.ul and qty > nutrient.ul:
                return False
        for ingredient, qty in self.ingredients_qty.items():
            if (
                ingredient.max_qty and qty > ingredient.max_qty
            ) or qty * ingredient.nutrients_qty.get(
                libnutrient.NUTRIENTS["Energy"], 0
            ) > MAX_INGREDIENT_CAL:
                return False
        return True

    def is_complete(self) -> bool:
        # all >= rdi and _check_omega_3_ratio
        return self.completeness_score >= 0.9999 and self.is_safe()
        for nutrient, qty in self.nutrients_qty.items():
            if qty < nutrient.rdi:
                return False
        return self._check_omega_3_ratio()

    def find_step_size(self, ingredient: libingredient.Ingredient) -> float:
        # This should be in the optimizer and should be settable to 1 (if usefulness) or 0 otherwise
        min_step_size = sys.maxsize
        # closest_nutrient = None  # dbg
        for nutrient, nutrient_qty_in_ingredient in ingredient.nutrients_qty.items():
            # find the step size needed to meet nutrient rdi
            if nutrient.rdi:
                step_size = (
                    (nutrient.rdi - self.nutrients_qty[nutrient]) / 1  # DBG
                ) / nutrient_qty_in_ingredient
                if step_size < min_step_size and step_size > 0:
                    min_step_size = step_size
                    # closest_nutrient = nutrient  # dbg

        if min_step_size == sys.maxsize:
            # if we need omega3 and this ingredient provides it, use it
            if not self._check_omega_3_ratio() and (
                ingredient.nutrients_qty.get(
                    libnutrient.NUTRIENTS["Omega-3 fatty acid"], 0
                )
                * 2
                > self.nutrients_qty.get(libnutrient.NUTRIENTS["Omega-6 fatty acid"], 0)
            ):
                min_step_size = self._find_ingredient_amt_to_reach_omega3_ratio(
                    ingredient
                )
                # return ingredient.step_qty
            else:
                return 0
        # else:
        #     return 1  # DBG

        # print(
        #     f"{ingredient=}, {max(min_step_size // 2, ingredient.step_qty)}, Closest nutrient: {closest_nutrient.name}"
        # )
        # TODO breakpoint if result is 1, for sunflower
        # if (
        #     max(int(min_step_size), ingredient.step_qty) <= 1
        #     and ingredient.name == "Vitamin D supplement"
        # ):
        #     breakpoint()
        # ) == 1 and closest_nutrient not in (
        #     libnutrient.NUTRIENTS["Vitamin A, RAE"],
        #     libnutrient.NUTRIENTS["Omega-3 fatty acid"],
        #     libnutrient.NUTRIENTS["Thiamin"],
        #     libnutrient.NUTRIENTS["Iodine, I"],
        #     libnutrient.NUTRIENTS["Vitamin C"],
        #     libnutrient.NUTRIENTS["Sodium, Na"],
        #     libnutrient.NUTRIENTS["Zinc, Zn"],
        #     libnutrient.NUTRIENTS["Vitamin K"],
        #     libnutrient.NUTRIENTS["Iron, Fe"],
        # ):
        # breakpoint()
        res = max(math.ceil(min_step_size), ingredient.step_qty)
        if libnutrient.NUTRIENTS["Energy"] in ingredient.nutrients_qty:
            calories_per_gram = ingredient.nutrients_qty[
                libnutrient.NUTRIENTS["Energy"]
            ]
            res = min(
                res,
                round(MAX_INGREDIENT_CAL / calories_per_gram)
                - self.ingredients_qty.get(ingredient, 0),
            )
        if ingredient.max_qty:
            return min(
                res, ingredient.max_qty - self.ingredients_qty.get(ingredient, 0)
            )
        return res

    def _find_ingredient_amt_to_reach_omega3_ratio(self, ingredient):
        omega3_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-3 fatty acid"]]
        omega6_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-6 fatty acid"]]
        omega3_qty_in_ingredient = ingredient.nutrients_qty.get(
            libnutrient.NUTRIENTS["Omega-3 fatty acid"], 0
        )
        omega6_qty_in_ingredient = ingredient.nutrients_qty.get(
            libnutrient.NUTRIENTS["Omega-6 fatty acid"], 0
        )
        if omega6_qty_in_ingredient == 0:
            return 1
        return (0.5 * omega6_qty - omega3_qty) / omega3_qty_in_ingredient

    def _get_omega_3_ratio(self) -> float:
        omega3_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-3 fatty acid"]]
        omega6_qty = self.nutrients_qty[libnutrient.NUTRIENTS["Omega-6 fatty acid"]]
        if omega6_qty == 0:
            return sys.maxsize if omega3_qty else 0
        return omega3_qty / omega6_qty

    def _check_omega_3_ratio(self) -> bool:
        return self._get_omega_3_ratio() >= OMEGA_36_RATIO

    def _is_ingredient_significant(self, ingredient: libingredient.Ingredient) -> bool:
        new_ingredients_qty = self.ingredients_qty.copy()
        new_ingredients_qty[ingredient] = 0
        new_recipe = Recipe(ingredients_qty=new_ingredients_qty)
        completeness_score = new_recipe.completeness_score
        # orig_qty = self.ingredients_qty[ingredient]
        # self.ingredients_qty[ingredient] = 0
        # completeness_score = self._compute_completeness_score(in_place=False)
        # if random.random() > 0.999:
        #     print(ingredient)
        #     print(orig_qty)
        #     print(completeness_score)
        #     print(self.completeness_score)
        #     breakpoint()
        # self.ingredients_qty[ingredient] = orig_qty
        return completeness_score < min(0.9995, self.completeness_score)

    def __lt__(self, other):
        return self.cost < other.cost
        # return self.loss < other.loss

    def __eq__(self, other):
        return self.ingredients_qty == other.ingredients_qty

    def __hash__(self):
        return hash(tuple(sorted(self.ingredients_qty.items())))

    def print(self, long=False):
        if long:
            for ingredient, qty in self.ingredients_qty.items():
                print(f"{ingredient.name}: {qty}")
            print("\n")
            for nutrient, qty in self.nutrients_qty.items():
                print(
                    f"{nutrient.name}: {qty}"
                    + (
                        f"{nutrient.unit}"
                        if not nutrient.rdi
                        else f" / {nutrient.rdi} {nutrient.unit} ({qty/nutrient.rdi*100:.2f}%)"
                    )
                )
            # print(
            #     {
            #         nutrient.name: (
            #             f"{qty}"
            #             + (
            #                 f"{nutrient.unit}"
            #                 if not nutrient.rdi
            #                 else f" / {nutrient.rdi} {nutrient.unit} ({qty/nutrient.rdi*100:.2f}%)"
            #             )
            #         )
            #         for nutrient, qty in self.nutrients_qty.items()
            #     }
            # )
        else:
            print(
                {
                    ingredient.name: qty
                    for ingredient, qty in self.ingredients_qty.items()
                }
            )
            print(
                {
                    nutrient.name: (
                        f"{qty:.2f}"
                        + (
                            f"{nutrient.unit}"
                            if not nutrient.rdi
                            else f" / {nutrient.rdi} {nutrient.unit} ({qty/nutrient.rdi*100:.2f}%)"
                        )
                    )
                    for nutrient, qty in self.nutrients_qty.items()
                }
            )
        print(f"{self.cost=:.2f}, {self.completeness_score=:.3f}, {self.n_steps=}")

    def save_to_yaml(self, fn: str):
        dict_to_save = {
            "optimization_cost": float(f"{float(self.cost):.2f}"),
            "cost": float(f"{float(self.calculate_true_cost()):.2f}"),
            "ingredients": {
                ingredient.name: float(f"{float(qty):.2f}")
                for ingredient, qty in self.ingredients_qty.items()
                if self._is_ingredient_significant(ingredient)
            },
            "nutrients": {},
            #     nutrient.name: float(f"{float(qty):.2f}")
            #     for nutrient, qty in self.nutrients_qty.items()
            # },
            "dry_weight_per_meal": float(
                f"{float(sum(self.ingredients_qty.values()))/3:.2f}"
            ),
            "cmd": " ".join(sys.argv),
        }

        def __find_main_source(
            nutrient: libnutrient.Nutrient,
            ingredients_qty: dict[libingredient.Ingredient, float],
        ) -> str:
            max_qty = 0
            insignificant_ingredients: list[libingredient.Ingredient] = []
            cmt = ""
            for ingredient, ingredient_qty in ingredients_qty.items():
                if nutrient in ingredient.nutrients_qty:
                    nutrient_qty = ingredient_qty * ingredient.nutrients_qty[nutrient]
                    if nutrient_qty > max_qty:
                        if not self._is_ingredient_significant(ingredient):
                            insignificant_ingredients.append((ingredient, nutrient_qty))
                            continue
                        max_qty = nutrient_qty
                        main_source = ingredient.name
            for (
                insignificant_ingredient,
                insignificant_ingredient_nutrient_qty,
            ) in insignificant_ingredients:
                if insignificant_ingredient_nutrient_qty >= max_qty:
                    print(
                        f"warning: {insignificant_ingredient.name} is the main source of {nutrient.name} but is insignificant"
                    )
                    cmt += f"({insignificant_ingredient.name}, {insignificant_ingredient_nutrient_qty:.2f})"
            return f"{main_source} ({max_qty:.2f} {nutrient.unit}{cmt})"
            # actually format max qty as 2 decimals

        # save nutrients[nutrient_name: {unit: qty, percent: qty/nutrient.rdi*100}, main_source: ingredient_name}
        for nutrient, nutrient_qty in self.nutrients_qty.items():
            if nutrient.rdi:
                dict_to_save["nutrients"][nutrient.name] = {
                    nutrient.unit: f"{nutrient_qty:.2f} / {nutrient.rdi} ({nutrient_qty/nutrient.rdi*100:.2f} %)",
                    # "percent rdi": float(f"{nutrient_qty/nutrient.rdi*100:.2f}"),
                    "main source": __find_main_source(nutrient, self.ingredients_qty),
                }
        # empty_ingredients = []
        # for ingredient, qty in dict_to_save["ingredients"].items():
        #     if qty == 0 or not self._is_ingredient_significant(ingredient):
        #         empty_ingredients.append(ingredient)
        # for ingredient in empty_ingredients:
        #     del dict_to_save["ingredients"][ingredient]
        with open(fn, "w", encoding="utf-8") as f:
            f.write(yaml.dump(dict_to_save, allow_unicode=True, width=float("inf")))

    @classmethod
    def load_from_yaml(
        cls, fpath: str, ingredients_list: list[libingredient.Ingredient]
    ):
        with open(fpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Convert ingredients list to a dictionary for quick access
        ingredients_dict = {
            ingredient.name: ingredient for ingredient in ingredients_list
        }
        ingredients_qty = {}

        # Load ingredients by name using the provided ingredients_list
        for ingredient_name, qty in data["ingredients"].items():
            if ingredient_name in ingredients_dict:
                ingredient_obj = ingredients_dict[ingredient_name]
                ingredients_qty[ingredient_obj] = qty
            else:
                raise ValueError(
                    f"Ingredient {ingredient_name} not found in provided ingredient list."
                )

        # Initialize a new Recipe object with the loaded ingredients
        new_recipe = cls(ingredients_qty=ingredients_qty)

        # # Load nutrients quantities from data, adjusting the new_recipe's state
        # for nutrient_name, qty in data["nutrients"].items():
        #     if nutrient_name in libnutrient.NUTRIENTS:
        #         nutrient_obj = libnutrient.NUTRIENTS[nutrient_name]
        #         if nutrient_obj in new_recipe.nutrients_qty:
        #             new_recipe.nutrients_qty[nutrient_obj] += qty
        #         else:
        #             new_recipe.nutrients_qty[nutrient_obj] = qty
        #     else:
        #         raise ValueError(f"Nutrient {nutrient_name} not found in libnutrient.NUTRIENTS.")

        # Manually compute cost and completeness score after all data is loaded
        new_recipe.cost = 0
        for ingredient, qty in ingredients_qty.items():
            new_recipe.cost += ingredient.calculate_cost(qty)
            for nutrient, nutrient_qty in ingredient.nutrients_qty.items():
                if nutrient in new_recipe.nutrients_qty:
                    new_recipe.nutrients_qty[nutrient] += qty * nutrient_qty
                else:
                    new_recipe.nutrients_qty[nutrient] = qty * nutrient_qty

        new_recipe._compute_completeness_score()

        return new_recipe
