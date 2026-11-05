from typing import List, Any, Optional
from subsystem3.optimisation_result import OptimisationResult
from subsystem3.exceptions_subsystem3 import VisualisationError

class ResultsVisualiser:
    """
    Responsible for generating visual comparisons (energy, carbon, cost, scores) 
    from optimisation results using Matplotlib or alternative plotting libraries.
    """

    def __init__(self, library: str = "matplotlib") -> None:
        self._library = library
        self._figure: Any = None

    # --- Getters and Setters ---
    def get_library(self) -> str:
        return self._library

    def set_library(self, library: str) -> None:
        self._library = library

    def get_figure(self) -> Any:
        return self._figure

    def set_figure(self, figure: Any) -> None:
        self._figure = figure

    # --- Visualisation Methods with Error Handling ---
    def plot_energy_comparison(self, results: List[OptimisationResult]) -> Any:
        """
        Generates an energy comparison chart across multiple optimisation results safely.
        """
        try:
            if not results:
                raise ValueError("Cannot plot energy comparison for an empty results list.")

            # Lazy import to prevent circular or missing display dependencies
            import matplotlib.pyplot as plt

            scenarios = [r.get_description() for r in results]
            energy_values = [r.get_energy_kwh() for r in results]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(scenarios, energy_values, color='skyblue', edgecolor='navy')
            ax.set_title("Energy Consumption Comparison by Scenario")
            ax.set_ylabel("Energy (kWh)")
            ax.set_xlabel("Scenario")
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()

            self._figure = fig
            return fig

        except Exception as e:
            raise VisualisationError(f"Failed to render energy comparison chart: {e}") from e

    def plot_cost_comparison(self, results: List[OptimisationResult]) -> Any:
        """
        Generates a cost comparison chart across multiple optimisation results safely.
        """
        try:
            if not results:
                raise ValueError("Cannot plot cost comparison for an empty results list.")

            import matplotlib.pyplot as plt

            scenarios = [r.get_description() for r in results]
            cost_values = [r.get_cost_eur() for r in results]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(scenarios, cost_values, color='lightgreen', edgecolor='darkgreen')
            ax.set_title("Operational Cost Comparison by Scenario")
            ax.set_ylabel("Cost (EUR)")
            ax.set_xlabel("Scenario")
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()

            self._figure = fig
            return fig

        except Exception as e:
            raise VisualisationError(f"Failed to render cost comparison chart: {e}") from e

    def plot_summary_scores(self, results: List[OptimisationResult]) -> Any:
        """
        Generates a summary score chart evaluating overall scenario effectiveness safely.
        """
        try:
            if not results:
                raise ValueError("Cannot plot summary scores for an empty results list.")

            import matplotlib.pyplot as plt

            scenarios = [r.get_description() for r in results]
            scores = [r.get_score() for r in results]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(scenarios, scores, marker='o', color='orange', linewidth=2, markersize=8)
            ax.set_title("Optimization Performance Scores")
            ax.set_ylabel("Score (0-100)")
            ax.set_xlabel("Scenario")
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()

            self._figure = fig
            return fig

        except Exception as e:
            raise VisualisationError(f"Failed to render summary scores chart: {e}") from e