from typing import Optional


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


NUTRIENTS = {
    nutrient_tuple[0]: Nutrient(*nutrient_tuple)
    for nutrient_tuple in (
        ("Energy", "kcal", 700, 2250),
        # ("Energy", "kcal", 700, 1600),  # TODO rm
        ("Carbohydrate", "g", 0, None),
        ("Protein", "g", 112, None),  # 1.6 g/kg * 70 kg
        ("Total lipid (fat)", "g", 0, None),
        ("Fatty acids, total saturated", "g", 0, None),
        ("Fatty acids, total monounsaturated", "g", 0, None),
        ("Fatty acids, total polyunsaturated", "g", 0, None),
        (
            "Omega-3 fatty acid",
            "g",
            3,
            None,
        ),  # Omega‐3 Polyunsaturated Fatty Acids Intake and Blood Pressure: A Dose‐Response Meta‐Analysis of Randomized Controlled Trials
        ("Omega-6 fatty acid", "g", 0, None),
        ("Fiber, total dietary", "g", 28, None),
        ("Cholesterol", "mg", 0, None),
        # Vitamin A is not dangerous from beta-carotene.
        # TODO warn user if recipe's vitamin A is over 3000
        (
            "Vitamin A, RAE",
            "µg",
            1350,
            3000,
        ),  # , 3000),  # NAM (2024)  # 900 * (3/2) s.t. 2 meals suffice
        ("Vitamin B-6", "mg", 1.7, 100),  # NAM, EFSA (2024)
        ("Vitamin B-12", "µg", 4.0, None),  # EFSA (2024)
        (
            "Vitamin C",
            "mg",
            330,
            2000,
        ),  # EFSA (2024) is 110 mg. More=better is not conclusive, but tripling s.t. RDI=1meal
        ("Vitamin D", "IU", 900, 4000),  # govt rec. = 600 IU., * 3/2 = 900 IU
        ("Vitamin E", "mg", 22.5, 300),  # NAM (2024) = 15, * 3/2 = 22.5
        ("Vitamin K", "µg", 180, None),  # NAM (2024) = 120, * 3/2 = 180
        ("Thiamin", "mg", 1.8, None),  # NAM (2024) = 1.2, * 3/2 = 1.8
        ("Riboflavin", "mg", 1.6, None),  # EFSA (2024) = 1.6, * 3/2 = 1.6
        ("Niacin", "mg", 24, 35),  # NAM (2024) = 16, * 3/2 = 24
        ("Folate", "µg", 600, 1000),  # NAM (2024) = 400, * 3/2 = 600
        ("Pantothenic acid", "mg", 7.5, None),  # NAM, EFSA (2024) = 5, * 3/2 = 7.5
        ("Biotin", "µg", 60, None),  # EFSA (2024) = 40, * 3/2 = 60
        ("Choline", "mg", 825, 3500),  # NAM (2024) = 550, * 3/2 = 825
        ("Calcium, Ca", "mg", 1500, 2500),  # NAM (2024) = 1000, * 3/2 = 1500
        ("Chloride", "g", 0, 3.6),  # 2.3, 3.6),  # not tracked by the USDA
        ("Chromium", "µg", 0, 35),  # 35, None),  # not tracked by the USDA
        ("Copper, Cu", "mg", 2.4, 5),  # EFSA (2024) = 1.6, * 3/2 = 2.4
        ("Iodine, I", "µg", 225, 1100),  # NAM, EFSA (2024) = 150, * 3/2 = 225
        ("Iron, Fe", "mg", 16.5, 45),  # EFSA (2024) = 11, * 3/2 = 16.5
        ("Magnesium, Mg", "mg", 630, None),  # NAM (2024) = 420, * 3/2 = 630
        ("Manganese, Mn", "mg", 4.5, 11),  # EFSA (2024) = 3, * 3/2 = 4.5
        ("Molybdenum, Mo", "µg", 0, 2000),  # 45, 2000),  # barely tracked by the USDA
        (
            "Phosphorus, P",
            "mg",
            1050,
            4000,
        ),  # US Institute of Medicine = 700, * 3/2 = 1050
        ("Potassium, K", "mg", 5250, None),  # EFSA (2024) = 3500, * 3/2 = 5250
        ("Selenium, Se", "µg", 105, 400),  # EFSA (2024) = 70, * 3/2 = 105
        ("Sodium, Na", "mg", 1750, 2300),  # between NAM's 1500 and EFSA's 2000
        ("Sulfur", "g", 0, None),  # 2, None), # barely tracked by the USDA
        (
            "Zinc, Zn",
            "mg",
            16.3,
            25,
        ),  # NAM: 11, EFSA: 9.4 to 16.3 depending on phytake intake
    )
}

NUTRIENTS["Vitamin A, RAE"]._conversion_functions[("IU", "µg")] = lambda qty: qty * 0.3
NUTRIENTS["Vitamin E"]._conversion_functions[("IU", "mg")] = lambda qty: qty * 0.67
NUTRIENTS["Energy"]._conversion_functions[("kJ", "kcal")] = lambda qty: qty / 4.184

if __name__ == "__main__":
    print(NUTRIENTS)
