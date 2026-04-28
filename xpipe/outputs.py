#!/usr/bin/env python3
"""
XPipe Experiment Outputs - Publication-Ready Results
Generates CSV, LaTeX tables, and high-quality plots
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentOutputs:
    """
    Manages all experiment outputs in human-readable formats.

    Features:
    - Schema-validated CSV files
    - Auto-generated LaTeX tables
    - Publication-quality plots (300 DPI PNG + PDF)
    - Organized directory structure
    """

    def __init__(self, exp_name: str, base_dir: str = "output"):
        self.exp_name = exp_name
        self.base_dir = Path(base_dir) / exp_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.base_dir / "metrics").mkdir(exist_ok=True)
        (self.base_dir / "figures").mkdir(exist_ok=True)
        (self.base_dir / "tables").mkdir(exist_ok=True)

    def save_metrics_csv(self, df: pd.DataFrame, name: str,
                        required_cols: Optional[list] = None) -> Path:
        """
        Save metrics to CSV with schema validation and metadata.

        Args:
            df: DataFrame with experimental results
            name: Filename (without .csv extension)
            required_cols: List of required column names for validation

        Returns:
            Path to saved CSV file
        """
        path = self.base_dir / "metrics" / f"{name}.csv"

        # Validate schema if specified
        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        # Add metadata header
        with open(path, 'w') as f:
            f.write(f"# Experiment: {self.exp_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Rows: {len(df)}\n")
            f.write(f"# Columns: {', '.join(df.columns)}\n")
            f.write("#\n")

        # Append data
        df.to_csv(path, mode='a', index=False, float_format='%.6f')

        try:
            rel_path = path.relative_to(Path.cwd())
            print(f" Saved CSV: {rel_path}")
        except ValueError:
            print(f" Saved CSV: {path}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

        return path

    def save_latex_table(self, df: pd.DataFrame, name: str,
                        caption: str, label: Optional[str] = None) -> Path:
        """
        Generate publication-ready LaTeX table.

        Args:
            df: DataFrame to convert
            name: Filename (without .tex extension)
            caption: Table caption
            label: LaTeX label (auto-generated if None)

        Returns:
            Path to saved .tex file
        """
        path = self.base_dir / "tables" / f"{name}.tex"

        if label is None:
            label = f"tab:{name}"

        # Generate LaTeX
        latex = df.to_latex(
            index=False,
            float_format="%.3f",
            caption=caption,
            label=label,
            escape=False,
            column_format='l' + 'c' * (len(df.columns) - 1),
            position='htbp'
        )

        # Add booktabs for professional formatting
        latex = latex.replace('\\toprule', '\\toprule\n').replace(
            '\\midrule', '\\midrule\n').replace('\\bottomrule', '\\bottomrule\n')

        path.write_text(latex)

        try:
            rel_path = path.relative_to(Path.cwd())
            print(f" Saved LaTeX table: {rel_path}")
        except ValueError:
            print(f" Saved LaTeX table: {path}")

        return path

    def save_plot(self, fig: plt.Figure, name: str,
                 dpi: int = 300, **kwargs) -> tuple[Path, Path]:
        """
        Save publication-quality plot in PNG and PDF formats.

        Args:
            fig: Matplotlib figure
            name: Filename (without extension)
            dpi: Resolution for PNG (default 300)
            **kwargs: Additional savefig arguments

        Returns:
            Tuple of (PNG path, PDF path)
        """
        png_path = self.base_dir / "figures" / f"{name}.png"
        pdf_path = self.base_dir / "figures" / f"{name}.pdf"

        # Save PNG (for presentations, web)
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', **kwargs)

        # Save PDF (for LaTeX inclusion)
        fig.savefig(pdf_path, bbox_inches='tight', **kwargs)

        print(f" Saved plot:")
        try:
            print(f"   PNG: {png_path.relative_to(Path.cwd())}")
            print(f"   PDF: {pdf_path.relative_to(Path.cwd())}")
        except ValueError:
            print(f"   PNG: {png_path}")
            print(f"   PDF: {pdf_path}")

        return png_path, pdf_path

    def create_summary_report(self, df: pd.DataFrame,
                             metrics: list[str]) -> Path:
        """
        Generate HTML summary report with key statistics.

        Args:
            df: Results DataFrame
            metrics: List of metric columns to summarize

        Returns:
            Path to HTML report
        """
        path = self.base_dir / "summary_report.html"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.exp_name} - Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{self.exp_name}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total samples: {len(df)}</p>

            <h2>Summary Statistics</h2>
            {df[metrics].describe().to_html()}

            <h2>Sample Data (first 10 rows)</h2>
            {df.head(10).to_html(index=False)}
        </body>
        </html>
        """

        path.write_text(html)
        try:
            print(f" Saved HTML report: {path.relative_to(Path.cwd())}")
        except ValueError:
            print(f" Saved HTML report: {path}")

        return path


if __name__ == "__main__":
    # Example usage
    import numpy as np

    outputs = ExperimentOutputs("example_experiment")

    # Example data
    df = pd.DataFrame({
        'dataset': ['tbmp', 'network'] * 10,
        'config': [f'cfg{i%5}' for i in range(20)],
        'quality': np.random.random(20),
        'latency_ms': np.random.randint(100, 2000, 20)
    })

    # Save CSV
    outputs.save_metrics_csv(df, "test_results",
                            required_cols=['dataset', 'quality'])

    # Save LaTeX table
    summary = df.groupby('dataset').agg({'quality': ['mean', 'std']})
    outputs.save_latex_table(summary, "test_summary",
                            caption="Example results summary")

    # Save plot
    fig, ax = plt.subplots()
    df.groupby('dataset')['quality'].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel('Quality')
    ax.set_title('Quality by Dataset')
    outputs.save_plot(fig, "test_plot")
    plt.close()

    print("\n Example outputs generated successfully!")
