import pandas as pd
from .data_profiler import extract_enhanced_metadata
from .chart_selector import ChartSelectorAgent
from aiapp.llm import get_llm_runner

class DashboardPipeline:
    def __init__(self):
        self.selector = ChartSelectorAgent(get_llm_runner())

    def run(self, df: pd.DataFrame, model_description: str = None) -> list[dict]:
        metadata = extract_enhanced_metadata(df)
        if model_description:
            metadata['model_description'] = model_description
        chart_suggestions = self.selector.suggest_charts(metadata)
        return chart_suggestions 