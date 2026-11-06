import matplotlib.pyplot as plt
from typing import List
from subsystem3.models import OptimisationResult

class ResultsVisualiser:
    """Visualises scenario comparison results using Matplotlib charts for the user dashboard."""
    
    def __init__(self, style: str = "default"):
        self.style = style

    def plot_energy_comparison(self, results: List[OptimisationResult]) -> None:
        """Generates a bar chart comparing energy consumption (kWh) across scenarios."""
        scenario_labels = [f"Scenario {i+1}" for i in range(len(results))]
        energy_values = [r.energy_kwh for r in results]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(scenario_labels, energy_values, color=['#4C72B0', '#55A868', '#C44E52'])
        plt.xlabel("Optimisation Scenarios", fontsize=12)
        plt.ylabel("Energy Consumption (kWh)", fontsize=12)
        plt.title("Scenario Comparison: Energy Consumption", fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_cost_comparison(self, results: List[OptimisationResult]) -> None:
        """Generates a bar chart comparing operational costs (EUR) across scenarios."""
        scenario_labels = [f"Scenario {i+1}" for i in range(len(results))]
        cost_values = [r.cost_eur for r in results]

        plt.figure(figsize=(8, 5))
        plt.bar(scenario_labels, cost_values, color=['#DD8452', '#CCB974', '#8172B2'])
        plt.xlabel("Optimisation Scenarios", fontsize=12)
        plt.ylabel("Cost (EUR)", fontsize=12)
        plt.title("Scenario Comparison: Operational Costs", fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_carbon_comparison(self, results: List[OptimisationResult]) -> None:
        """Generates a bar chart comparing carbon emissions (kg CO2) across scenarios."""
        scenario_labels = [f"Scenario {i+1}" for i in range(len(results))]
        carbon_values = [r.carbon_kg for r in results]

        plt.figure(figsize=(8, 5))
        plt.bar(scenario_labels, carbon_values, color=['#64B5CD', '#8172B2', '#937860'])
        plt.xlabel("Optimisation Scenarios", fontsize=12)
        plt.ylabel("Carbon Emissions (kg CO2)", fontsize=12)
        plt.title("Scenario Comparison: Carbon Footprint", fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()