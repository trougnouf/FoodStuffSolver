import queue
import time
import sys
from typing import Union, Optional
import argparse
import random

import yaml
import cma

# You may need to install numpy and scipy: pip install numpy scipy
import numpy as np
from scipy.optimize import linprog

import libingredient
import libnutrient
import librecipe

# Shared constant for max calories from a single ingredient
MAX_INGREDIENT_CAL = 400

# TODO add calories target (or nutrient override)
# TODO add is_significant to ingredient wrt quantity and nutrients (eg adding up to 1% of any nutrient is significant)
# TODO check that allergies exist / enum?
# TODO implement ingredient expiration (increases cost according to usage)
# TODO penalize ingredients' nutrient after they've provided 75% of the RDI

class Solver:
    def _make_base_recipe(self, init_ingredients_str_qty: Optional[dict[str, float]]):
        base_ingredients_qty = {ingredient: 0 for ingredient in self.ingredients}
        if init_ingredients_str_qty:
            for ingredient_str, qty in init_ingredients_str_qty.items():
                for ingredient in self.ingredients:
                    if ingredient.name == ingredient_str:
                        base_ingredients_qty[ingredient] = qty
                        break
                else:
                    raise ValueError(f"Unknown ingredient {ingredient_str}")
        return librecipe.Recipe(base_ingredients_qty)

    def _compute_min_cost_of_nutrients(self) -> None:
        self.nutrients_min_cost = {}
        nutrient_min_cost_src = {}
        for ingredient in self.ingredients:
            for nutrient, nutrient_qty in ingredient.nutrients_qty.items():
                if nutrient_qty == 0:
                    continue
                nutrient_cost = ingredient.cost_per_unit / nutrient_qty
                if (
                    nutrient not in self.nutrients_min_cost
                    or nutrient_cost < self.nutrients_min_cost[nutrient]
                ):
                    self.nutrients_min_cost[nutrient] = nutrient_cost
                    nutrient_min_cost_src[nutrient] = ingredient

    def _preload_recipe(self, fpath: str) -> librecipe.Recipe:
        return librecipe.Recipe.load_from_yaml(
            fpath=fpath, ingredients_list=self.ingredients
        )

    def __init__(
        self,
        ingredients: list[libingredient.Ingredient],
        init_ingredients_str_qty: Optional[dict[str, float]] = None,
    ):
        self.ingredients = ingredients
        self._compute_min_cost_of_nutrients()


class CMASolver(Solver):
    def __init__(
        self,
        ingredients,
        init_ingredients_str_qty: Optional[dict[str, float]] = None,
        speedup_factor: int = 2000,
        preload_recipe_fpath: Optional[str] = None,
    ):
        super().__init__(ingredients, init_ingredients_str_qty)
        self.ingredients = ingredients
        upper_bounds = [
            (
                ingredient.max_qty
                or MAX_INGREDIENT_CAL
                / ingredient.nutrients_qty.get(libnutrient.NUTRIENTS["Energy"], 0.1)
            )
            for ingredient in self.ingredients
        ]
        lower_bounds = [0] * len(self.ingredients)

        if preload_recipe_fpath:
            print(f"Loading preloaded recipe from {preload_recipe_fpath} to set starting point for CMA-ES.")
            preloaded_recipe = self._preload_recipe(preload_recipe_fpath)
            for i, ingredient in enumerate(self.ingredients):
                if ingredient in preloaded_recipe.ingredients_qty:
                    # For CMA, we use the preloaded values as the starting point (the initial mean)
                    lower_bounds[i] = preloaded_recipe.ingredients_qty[ingredient]

        # CMA-ES's second argument is the initial standard deviation (step size).
        # The first argument is the initial mean (starting point).
        self.es = cma.CMAEvolutionStrategy(
            lower_bounds.copy(), 0.5, {"bounds": [[0]*len(self.ingredients), upper_bounds]}
        )
        self.ignore_future_estimate = False
        self.speedup_factor = speedup_factor
        self.best_solution = None

    def _compute_greedy_min_cost_to_complete(self, recipe: librecipe.Recipe) -> float:
        if not recipe.is_safe():
            return sys.maxsize
        highest_cost = 0
        for nutrient, nutrient_qty in recipe.nutrients_qty.items():
            if nutrient_qty < nutrient.rdi:
                cost = self.nutrients_min_cost[nutrient] * (nutrient.rdi - nutrient_qty)
                if cost > highest_cost:
                    highest_cost = cost
        return highest_cost

    def _compute_loss(self, solution: list[float]) -> float:
        ingredients_qty = {
            ingredient: solution[i] for i, ingredient in enumerate(self.ingredients)
        }
        recipe = librecipe.Recipe(ingredients_qty)
        res = (
            recipe.cost
            + (
                0
                if self.ignore_future_estimate
                else self._compute_greedy_min_cost_to_complete(recipe)
            )
            + (
                self.speedup_factor
                - recipe.completeness_score**2
                * (self.speedup_factor / (1 if recipe.is_complete() else 1.5))
            )
        )
        if recipe.is_complete():
            if self.best_solution is None or recipe.cost < self.best_solution.cost:
                self.best_solution = recipe
                print("\n--- New Best Solution Found ---")
                recipe.print(long=True)
                recipe.save_to_yaml(f"recipe_cma.yaml")
        if random.random() > 0.9995:
            print("Random sample print:")
            recipe.print(long=False)
        return res

    def solve(self):
        self.es.optimize(self._compute_loss)
        return self.best_solution


class LPSolver(Solver):
    def __init__(
        self, 
        ingredients, 
        preload_recipe_fpath: Optional[str] = None,
        max_calories: Optional[float] = None,
        min_proteins: Optional[float] = None,
        **kwargs
    ):
        super().__init__(ingredients)
        self.ingredients = sorted(ingredients, key=lambda i: i.name)
        self.preload_recipe_fpath = preload_recipe_fpath
        self.max_calories = max_calories
        self.min_proteins = min_proteins
    
    def solve(self) -> Union[librecipe.Recipe, None]:
        # This function will now automatically use the globally loaded NUTRIENTS
        # No changes are needed here.
        print("Setting up the Linear Programming problem...")
        if self.max_calories is not None:
            print(f"--> Overriding max calories to: {self.max_calories} kcal")
        if self.min_proteins is not None:
            print(f"--> Overriding min protein to: {self.min_proteins} g")
        
        c = np.array([ing.calculate_cost(1, true_cost=True) for ing in self.ingredients])
        constraints_A, constraints_b = [], []
        sorted_nutrients = sorted(libnutrient.NUTRIENTS.values(), key=lambda n: n.name)

        for nutrient in sorted_nutrients:
            nutrient_vector = np.array([ing.nutrients_qty.get(nutrient, 0) for ing in self.ingredients])
            target_rdi = self.min_proteins if nutrient.name == "Protein" and self.min_proteins is not None else nutrient.rdi
            target_ul = self.max_calories if nutrient.name == "Energy" and self.max_calories is not None else nutrient.ul
            
            if target_rdi > 0:
                constraints_A.append(-nutrient_vector)
                constraints_b.append(-target_rdi)
            if target_ul is not None:
                constraints_A.append(nutrient_vector)
                constraints_b.append(target_ul)

        omega3 = libnutrient.NUTRIENTS["Omega-3 fatty acid"]
        omega6 = libnutrient.NUTRIENTS["Omega-6 fatty acid"]
        omega3_vector = np.array([ing.nutrients_qty.get(omega3, 0) for ing in self.ingredients])
        omega6_vector = np.array([ing.nutrients_qty.get(omega6, 0) for ing in self.ingredients])
        ratio_vector = omega3_vector - (librecipe.OMEGA_36_RATIO * omega6_vector)
        constraints_A.append(-ratio_vector)
        constraints_b.append(0)

        A_ub = np.array(constraints_A)
        b_ub = np.array(constraints_b)

        # Bounds logic remains the same
        bounds = []
        preloaded_quantities = {}
        if self.preload_recipe_fpath:
            preloaded_recipe = self._preload_recipe(self.preload_recipe_fpath)
            preloaded_quantities = preloaded_recipe.ingredients_qty
        
        for ing in self.ingredients:
            lower_bound = preloaded_quantities.get(ing, 0)
            energy_per_g = ing.nutrients_qty.get(libnutrient.NUTRIENTS["Energy"], 0.1)
            max_cal_qty = MAX_INGREDIENT_CAL / energy_per_g if energy_per_g > 0 else float('inf')
            upper_bound = min(ing.max_qty, max_cal_qty) if ing.max_qty else max_cal_qty
            if lower_bound > upper_bound:
                raise ValueError(f"Min quantity for {ing.name} ({lower_bound}g) exceeds max ({upper_bound:.2f}g).")
            bounds.append((lower_bound, upper_bound))

        print("Solving with scipy.optimize.linprog (method='highs')...")
        result = linprog(np.array(c), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not result.success:
            print(f"\n--- Solver failed: {result.message} ---")
            return None
        
        ingredients_qty = {
            self.ingredients[i]: result.x[i] for i in range(len(self.ingredients)) if result.x[i] > 1e-6
        }
        return librecipe.Recipe(ingredients_qty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A nutritionally complete recipe solver.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=['lp', 'cma'],
        default='lp',
        help="The solver to use. 'lp' is fast and optimal."
    )
    parser.add_argument(
        "--nutrient_profile",
        type=str,
        default=None,
        help="Path to a YAML file defining nutrient RDIs and ULs. If not provided, defaults are used."
    )
    parser.add_argument(
        "--ingredients_dpaths", type=str, nargs="+", default=[libingredient.INGREDIENTS_DIR],
        help=f"Path to ingredients directory (default: {libingredient.INGREDIENTS_DIR})",
    )
    parser.add_argument(
        "--allergies", nargs="+", default=[], help="List of allergies to exclude",
    )
    parser.add_argument(
        "--max_calories", type=float, default=None, help="Optional: Override max daily calories (for LP solver)."
    )
    parser.add_argument(
        "--min_proteins", type=float, default=None, help="Optional: Override min daily protein in grams (for LP solver)."
    )
    parser.add_argument(
        "--preload_recipe_fpath", type=str, default=None, help="Path to a YAML recipe to enforce minimum quantities (LP)."
    )
    # (other args for CMA etc.)
    parser.add_argument("--speedup_factor", type=int, default=2000)
    parser.add_argument("--ingredients_augmentations", type=str, default=None)
    args = parser.parse_args()

    # --- INITIALIZE NUTRIENTS ---
    # This must be done *before* any other part of the code that relies on libnutrient.NUTRIENTS
    libnutrient.initialize_nutrients(fpath=args.nutrient_profile)

    ingredients = libingredient.get_ingredients(dpaths=args.ingredients_dpaths, allergies=args.allergies)
    if args.ingredients_augmentations:
        with open(args.ingredients_augmentations, "r") as f:
            augmentations = yaml.safe_load(f)
        libingredient.augment_ingredients(ingredients, augmentations)

    solver = None
    if args.solver == 'lp':
        solver = LPSolver(
            ingredients, 
            preload_recipe_fpath=args.preload_recipe_fpath,
            max_calories=args.max_calories,
            min_proteins=args.min_proteins
        )
    elif args.solver == 'cma':
        solver = CMASolver(
            ingredients, 
            speedup_factor=args.speedup_factor,
            preload_recipe_fpath=args.preload_recipe_fpath
        )
    
    if solver:
        recipe = solver.solve()
        if recipe:
            print("\n--- Solver Finished ---")
            recipe.print(long=True)
            output_filename = f"recipe_{args.solver}.yaml"
            recipe.save_to_yaml(output_filename)
            print(f"\nRecipe saved to {output_filename}")
        else:
            print("\nNo feasible recipe was found.")
