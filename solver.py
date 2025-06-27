import queue
import time
import sys
from typing import Union, Optional

import yaml
import libingredient
import libnutrient
import librecipe
import cma
import random
import argparse

# TODO add calories target (or nutrient override)
# TODO add is_significant to ingredient wrt quantity and nutrients (eg adding up to 1% of any nutrient is significant)
# TODO check that allergies exist / enum?
# TODO implement ingredient expiration (increases cost according to usage)
# TODO penalize ingredients' nutrient after they've provided 75% of the RDI

# class AbstractSolver:
#     def __init__(self, ingredients):
#         self.ingredients = ingredients


class Solver:
    def _make_base_recipe(init_ingredients_str_qty: Optional[dict[str, float]]):
        base_ingredients_qty = {ingredient: 0 for ingredient in self.ingredients}
        if init_ingredients_str_qty:
            for ingredient_str, qty in init_ingredients_str_qty.items():
                for ingredient in ingredients:
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
                # nutrient_cost = max(
                #     nutrient_cost, ingredient.cost_per_unit * ingredient.min_cost_step
                # )
                if (
                    nutrient not in self.nutrients_min_cost
                    or nutrient_cost < self.nutrients_min_cost[nutrient]
                ):
                    self.nutrients_min_cost[nutrient] = nutrient_cost
                    nutrient_min_cost_src[nutrient] = ingredient

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


class NaiveSolver(Solver):
    def __init__(
        self,
        ingredients: list[libingredient.Ingredient],
        # base_recipe: Optional[librecipe.Recipe] = None,
        init_ingredients_str_qty: Optional[dict[str, float]] = None,
        speedup_factor: int = 30,
        ignore_future_estimate=False,
    ):
        super().__init__(ingredients, init_ingredients_str_qty)
        base_recipe = self._make_base_recipe(init_ingredients_str_qty)
        self.known_recipes_hashes = set()
        self.known_recipes_hashes.add(
            hash(tuple(sorted(base_recipe.ingredients_qty.items())))
        )
        self.recipes_to_explore = queue.PriorityQueue()
        self.recipes_to_explore.put((self._compute_loss(base_recipe), base_recipe))
        self.backlog = queue.PriorityQueue()

        self.n_explored_recipes = 0

        self.speedup_factor = speedup_factor
        self.ignore_future_estimate = ignore_future_estimate

        self.known_recipes_hashes.add(
            hash(tuple(sorted(base_recipe.ingredients_qty.items())))
        )
        self.recipes_to_explore = queue.PriorityQueue()
        self.recipes_to_explore.put((self._compute_loss(base_recipe), base_recipe))
        self.backlog = queue.PriorityQueue()

        self.n_explored_recipes = 0

        # print("Most efficient nutrient sources:")
        # for nutrient in self.nutrients_min_cost:

        #     print(f"{nutrient.name}: {nutrient_min_cost_src[nutrient].name}")
        # print("\n")

    def _compute_loss(self, recipe: librecipe.Recipe) -> float:
        return (
            recipe.cost
            + (
                0
                if self.ignore_future_estimate
                else self._compute_greedy_min_cost_to_complete(recipe)
            )
            + (
                self.speedup_factor
                - recipe.completeness_score**2 * self.speedup_factor
                # - recipe.completeness_score * self.speedup_factor
            )
        )

    def solve(self) -> librecipe.Recipe:
        last_stats_print = 0
        last_recipe_print = 0
        init_time = time.time()
        best_full_recipe = None
        while not (self.recipes_to_explore.empty() and self.backlog.empty()):
            if self.recipes_to_explore.empty():
                self.recipes_to_explore, self.backlog = (
                    self.backlog,
                    self.recipes_to_explore,
                )
            loss, recipe = self.recipes_to_explore.get()
            # self.known_recipes.add(recipe)
            if best_full_recipe is not None and recipe.cost > best_full_recipe.cost:
                continue
            if recipe.is_complete():
                print(f"Found a complete recipe with {recipe.cost=}")
                recipe = self.reduce_cost(recipe)
                print(f"Tried to reduce cost. {recipe.cost=}")
                if not best_full_recipe or recipe.cost < best_full_recipe.cost:
                    best_full_recipe = recipe
                    recipe.save_to_yaml("best_recipe_ex3.yaml")
                    recipe.print(long=True)
                continue
                return recipe
            current_additional_recipes = queue.PriorityQueue()
            for ingredient in self.ingredients:
                # new_ingredients_qty = recipe.ingredients_qty.copy()
                # new_ingredients_qty[ingredient] += ingredient.step_qty
                new_step_size = recipe.find_step_size(ingredient)
                new_recipe = recipe.expand(ingredient, new_step_size)
                # new_ingredients_qty[ingredient] += new_step_size
                new_ingredients_qty_static_hash = hash(
                    tuple(sorted(new_recipe.ingredients_qty.items()))
                )
                if new_ingredients_qty_static_hash in self.known_recipes_hashes:
                    continue
                assert new_step_size > 0  # should have continued if 0
                # TODO check if recipe is an improvement (?)
                self.known_recipes_hashes.add(new_ingredients_qty_static_hash)
                # new_recipe = librecipe.Recipe(
                #     new_ingredients_qty, n_steps=recipe.n_steps + 1
                # )
                # if new_recipe not in self.known_recipes and new_recipe.is_safe():
                if new_recipe.is_safe() and (
                    best_full_recipe is None or new_recipe.cost < best_full_recipe.cost
                ):
                    # self.recipes_to_explore.put(
                    current_additional_recipes.put(
                        (
                            self._compute_loss(new_recipe),
                            new_recipe,
                        )
                    )
            for _ in range(2):
                if current_additional_recipes.empty():
                    break
                self.recipes_to_explore.put(current_additional_recipes.get())
            while not current_additional_recipes.empty():
                self.backlog.put(current_additional_recipes.get())
            self.n_explored_recipes += 1
            if last_stats_print + 1000 <= self.n_explored_recipes:
                print(f"Explored {self.n_explored_recipes} recipes")
                print(f"Recipes to explore: {self.recipes_to_explore.qsize()}")
                # print recipe cost, completeness_score, and loss with 2 decimals
                print(
                    f"Current {recipe.cost=:.2f}, {recipe.completeness_score=:.3f}, {loss=:.2f}, best full recipe cost: {404 if not best_full_recipe else best_full_recipe.cost:.2f}"
                )
                last_stats_print = self.n_explored_recipes
                if last_recipe_print + 10000 <= self.n_explored_recipes:
                    print("Current recipe:")
                    recipe.print()
                    last_recipe_print = self.n_explored_recipes
                    print(f"Time elapsed: {time.time() - init_time}")
                    recipe.save_to_yaml("current_recipe.yaml")

    def reduce_cost(self, recipe: librecipe.Recipe) -> librecipe.Recipe:
        while True:
            starting_cost = recipe.cost
            for ingredient, qty in dict(
                sorted(
                    recipe.ingredients_qty.items(),
                    key=lambda ingredient_qty: ingredient_qty[0].cost_per_unit,
                    reverse=True,
                )
            ).items():
                if qty > 0:
                    new_ingredients_qty = recipe.ingredients_qty.copy()
                    new_ingredients_qty[ingredient] -= 1
                    new_recipe = librecipe.Recipe(new_ingredients_qty)
                    if (
                        new_recipe.is_safe()
                        and new_recipe.is_complete()
                        and new_recipe.cost < recipe.cost
                    ):
                        recipe = new_recipe
                        break
            if recipe.cost == starting_cost:
                break
        return recipe


class CMASolver(Solver):
    MAX_INGREDIENT_CAL = 400  # avoid using an ingredient for more than 400 kcal per day

    def __init__(
        self,
        ingredients,
        init_ingredients_str_qty: Optional[dict[str, float]] = None,
        speedup_factor: int = 2000,
        preload_recipe_fpath: Optional[str] = None,
    ):
        self.ingredients = ingredients
        upper_bounds = [
            (
                ingredient.max_qty
                or self.MAX_INGREDIENT_CAL
                / ingredient.nutrients_qty.get(libnutrient.NUTRIENTS["Energy"], 0.1)
            )
            # * 0.9  # HACK for all nutrients w/ 2/3 calories
            for ingredient in self.ingredients
        ]
        lower_bounds = [0] * len(self.ingredients)

        print(lower_bounds)
        print(upper_bounds)
        if preload_recipe_fpath:
            preloaded_recipe = self._preload_recipe(preload_recipe_fpath)
            for i, ingredient in enumerate(self.ingredients):
                if ingredient in preloaded_recipe.ingredients_qty:
                    lower_bounds[i] = min(
                        preloaded_recipe.ingredients_qty[ingredient],
                        upper_bounds[i] * 0.99,
                    )
                    assert lower_bounds[i] < upper_bounds[i]
        self.es = cma.CMAEvolutionStrategy(
            lower_bounds.copy(), 0.5, {"bounds": [lower_bounds, upper_bounds]}
        )
        self._compute_min_cost_of_nutrients()
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
        ingredients_qty = {}
        for i, ingredient in enumerate(self.ingredients):
            ingredients_qty[ingredient] = solution[i]
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
                * (self.speedup_factor / (1 if recipe.is_complete else 1.5))
                # - recipe.completeness_score * self.speedup_factor
            )
        )
        if recipe.is_complete():
            if self.best_solution is None or recipe.cost < self.best_solution.cost:
                self.best_solution = recipe
                recipe.print(long=True)
                recipe.save_to_yaml(
                    f"cma_{self.speedup_factor}_2000cal.yaml"
                )  # TODO _mk_fn(self) and add allergens to fn
        if random.random() > 0.9995:
            print("random print:")
            recipe.print(long=False)
        return res

    def solve(self):
        def optimize():
            while not self.es.stop():
                solutions = self.es.ask()
                self.es.tell(solutions, [self._compute_loss(s) for s in solutions])
                self.es.logger.add()
                self.es.disp()
            return self.es.result.xbest

        sol = optimize()
        print(sol)
        return sol


"""
Future solver idea:

compute the current + future cost of cheapest ingredient
    for each ingredient: compute actual+future cost and add best 3 ingredients to todo list

make ingredient initialization based on current one: add 1 unit of ingredient to current recipe and compute costs instead of recomputing all

# TODO params:
- set base recipe
- set custom nutrient min/max values
- set output fn
"""

"""Parse argument for ingredients_dpath and list of allergies."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingredients_dpaths",
        type=str,
        nargs="+",
        default=[libingredient.INGREDIENTS_DIR],
        help=f"Path(s) to ingredients directory·ies (default: {libingredient.INGREDIENTS_DIR})",
    )
    parser.add_argument(
        "--allergies",
        nargs="+",
        default=[],
        help="List of allergies (default: [])",
    )
    parser.add_argument(
        "--speedup_factor",
        type=int,
        default=2000,
        help="Speedup factor (default: 2000)",
    )
    parser.add_argument(
        "--ingredients_augmentations",
        type=str,
        default=None,
        help="Path to a yaml file containing extra parameters about some ingredients (eg satisfaction_multiplier)",
    )
    parser.add_argument("--preload_recipe_fpath", type=str, default=None)
    args = parser.parse_args()

    ingredients = libingredient.get_ingredients(
        dpaths=args.ingredients_dpaths, allergies=args.allergies
    )
    if args.ingredients_augmentations:
        with open(args.ingredients_augmentations, "r") as f:
            augmentations = yaml.safe_load(f)
        libingredient.augment_ingredients(ingredients, augmentations)
    solver = CMASolver(ingredients, preload_recipe_fpath=args.preload_recipe_fpath)
    recipe = solver.solve()
    print("2nd pass")
    recipe = solver.solve()
    # recipe.print(long=True)
    print(recipe)
    # recipe.save_to_yaml("recipe_cma.yaml")
