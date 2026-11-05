import matplotlib.pyplot as plt
import pandas as pd
from subsystem3.optimisation_result import OptimisationResult

class ResultsVisualiser:
    """Visualises optimisation scenario comparisons using Matplotlib charts."""
    
    def __init__(self, library: str = "matplotlib"):
        self.library = library

    def plot_energy_comparison(self, results: list[OptimisationResult]):
        descriptions = [r.description for r in results]
        energies = [r.energy_kwh for r in results]
        
        plt.figure(figsize=(8, 4))
        plt.barh(descriptions, energies, color='skyblue')
        plt.xlabel("Total Energy Consumption (kWh)")
        plt.title("Scenario Energy Comparison")
        plt.tight_layout()
        plt.savefig("energy_comparison.png")
        plt.close()

    def plot_cost_comparison(self, results: list[OptimisationResult]):
        descriptions = [r.description for r in results]
        costs = [r.cost_eur for r in results]
        
        plt.figure(figsize=(8, 4))
        plt.barh(descriptions, costs, color='lightgreen')
        plt.xlabel("Total Cost (EUR)")
        plt.title("Scenario Cost Comparison")
        plt.tight_layout()
        plt.savefig("cost_comparison.png")
        plt.close()