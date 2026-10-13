import numpy as np
import pandas as pd


class SilicaSandEconomicEngine:
    """
    Macroeconomic Value-Addition Engine for Egypt's High-Purity Silica Sand.
    Models industrial conversion of raw sand into float & solar panel glass vs raw export.
    """

    def __init__(self, base_gdp_usd=380e9):
        """
        Args:
            base_gdp_usd (float): Baseline Annual GDP of Egypt in USD ($380 Billion USD).
        """
        self.base_gdp_usd = base_gdp_usd

    def calculate_value_add(self, raw_sand_tonnage=3_500_000, raw_price_per_ton=25.0,
                             glass_yield_ratio=0.80, glass_price_per_ton=450.0):
        """
        Calculates annual financial returns and Egyptian GDP contribution percentage.
        """
        # 1. Raw Sand Export Scenario
        raw_revenue = raw_sand_tonnage * raw_price_per_ton

        # 2. Industrial Glass Manufacturing Scenario
        processed_glass_tonnage = raw_sand_tonnage * glass_yield_ratio
        glass_revenue = processed_glass_tonnage * glass_price_per_ton

        # 3. Macroeconomic Gains
        net_value_add = glass_revenue - raw_revenue
        value_multiplier = glass_revenue / raw_revenue if raw_revenue > 0 else 0.0
        gdp_boost_percentage = (net_value_add / self.base_gdp_usd) * 100.0

        return {
            "raw_sand_tonnage": raw_sand_tonnage,
            "raw_revenue_usd": raw_revenue,
            "processed_glass_tonnage": processed_glass_tonnage,
            "glass_revenue_usd": glass_revenue,
            "net_value_add_usd": net_value_add,
            "value_multiplier": value_multiplier,
            "gdp_boost_percentage": gdp_boost_percentage,
            "base_gdp_usd": self.base_gdp_usd
        }

    def generate_10year_projection(self, initial_tonnage=3_500_000, annual_growth_rate=0.05):
        """
        Generates a 10-year comparative trajectory DataFrame (Raw Export vs Glass Industry).
        """
        years = np.arange(2026, 2036)
        raw_revs = []
        glass_revs = []
        net_gains = []

        curr_tonnage = initial_tonnage
        for _ in years:
            metrics = self.calculate_value_add(raw_sand_tonnage=curr_tonnage)
            raw_revs.append(metrics["raw_revenue_usd"] / 1e6)      # $ Millions
            glass_revs.append(metrics["glass_revenue_usd"] / 1e6)  # $ Millions
            net_gains.append(metrics["net_value_add_usd"] / 1e6)   # $ Millions
            curr_tonnage *= (1.0 + annual_growth_rate)

        df = pd.DataFrame({
            "Year": years,
            "Raw Sand Export ($M)": np.round(raw_revs, 2),
            "Manufactured Glass Revenue ($M)": np.round(glass_revs, 2),
            "Net Value Addition ($M)": np.round(net_gains, 2)
        })
        return df
