# FoodStuffSolver

A Python-based tool that generates cost-effective, nutritionally complete recipes using Linear Programming (LP) or CMA-ES evolutionary strategies. It balances macronutrients, micronutrients, and specific ratios (like Omega-3/6) while minimizing cost.

## Dependencies

```bash
pip install pyyaml numpy scipy cma requests
```

## Setup & Data

### 1. Importing Ingredients (USDA)
To fetch nutrient data from the USDA FoodData Central database:

1.  Get an API key from [USDA API Sign Up](https://fdc.nal.usda.gov/api-key-signup.html).
2.  Save the key in a file named `usda_api.key` in the root directory.
3.  Run the tool with a Food ID:
    ```bash
    python tools/get_usda_ingredient.py <USDA_FOOD_ID>
    ```

### 2. Manual Editing (Required)
Ingredients imported from the USDA **do not** contain price or allergy information. You must manually edit the generated YAML files in the `ingredients/` directory:

*   **`cost_per_unit`**: Enter the cost per unit (default is per **gram**). The solver treats ingredients with no cost as free/infinite, which may distort results.
*   **`allergen`**: Add a list of allergens if applicable (e.g., `allergen: ["Peanuts"]`).

## Usage

### Generating a Recipe
Run the solver to generate an optimal recipe based on your `ingredients/` folder.

**Basic Usage (Linear Programming - Fast/Optimal):**
```bash
python solver.py
```

**Common Options:**
```bash
python solver.py \
  --max_calories 2000 \
  --min_proteins 120 \
  --allergies "Peanuts" "Shellfish" \
  --ignore_max_qty
```

### Key Arguments
| Argument | Description |
| :--- | :--- |
| `--solver` | `lp` (default, fast) or `cma` (evolutionary, slower). |
| `--allergies` | List of allergens to exclude (checks the `allergen` field in YAMLs). |
| `--max_calories` | Override maximum daily energy (kcal). |
| `--min_proteins` | Override minimum daily protein (g). |
| `--min_nutrient` | Set specific min for a nutrient (e.g., `--min_nutrient "Iron, Fe" 20`). |
| `--ingredients_dpaths`| Custom path to ingredients directory (default: `ingredients`). |

## Configuration

*   **Nutrient Profiles:** `libnutrient.py` contains default RDI/UL definitions. You can provide a custom profile via `--nutrient_profile my_profile.yaml`.
*   **Ingredient Attributes:** Aside from cost and allergens, you can also modify `max_qty` (limit usage of specific items) or `satisfaction_multiplier` in the ingredient YAML files.

## File Structure
*   `solver.py`: Main entry point.
*   `libingredient.py`: Ingredient data models and I/O.
*   `librecipe.py`: Logic for recipe aggregation, cost calculation, and validation.
*   `libnutrient.py`: Nutrient definitions and unit conversions.
*   `tools/`: Scripts for data fetching.
