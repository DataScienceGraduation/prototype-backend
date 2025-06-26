import json

class ChartSelectorAgent:
    def __init__(self, llm_runner):
        self.llm_runner = llm_runner

    def suggest_charts(self, metadata: dict) -> list[dict]:
        plotly_types = [
            "histogram", "scatter", "bar", "line", "box", "pie", "violin", "heatmap"
        ]
        model_description = metadata.get('model_description')
        prompt = (
            (f"Model description (for context):\n{model_description}\n\n" if model_description else "") +
            "Given this dataset metadata:\n"
            f"{json.dumps(metadata, indent=2, default=str)}\n\n"
            "Suggest 3-5 essential visualizations considering:\n"
            "1. Data types and distributions (skew/kurtosis)\n"
            "2. Missing value percentages\n"
            "3. Top correlations\n"
            f"4. Dataset size ({metadata['overview']['num_rows']} rows, {metadata['overview']['num_cols']} cols)\n\n"
            "Prioritize charts that:\n"
            "- Reveal data quality issues\n"
            "- Show key distributions\n"
            "- Highlight important relationships\n\n"
            "Respond ONLY in JSON format.\n"
            "chart_type MUST be one of: histogram, scatter, bar, line, box, pie, violin, heatmap.\n"
            "Example:\n"
            "{\n  \"charts\": [\n    {\n      \"chart_type\": \"histogram\",\n      \"columns\": [\"age\"],\n      \"insight\": \"Show distribution of age with skew=0.4\",\n      \"priority\": \"high\"\n    }\n  ]\n}\n"
        )
        raw = self.llm_runner(prompt)
        print("LLM raw output:", repr(raw))
        if not raw or not raw.strip():
            raise ValueError("LLM returned empty response for chart suggestions.")
        if raw.strip().startswith("```"):
            raw = raw.strip().strip("`").strip("json").strip()
        try:
            charts = json.loads(raw)["charts"]
            # Enforce only plotly-supported types
            filtered = [c for c in charts if c.get("chart_type") in plotly_types]
            if not filtered:
                raise ValueError(f"No supported chart types found in LLM output: {charts}")
            return filtered
        except Exception as e:
            raise ValueError(f"Failed to parse chart suggestions JSON from LLM: {raw}") from e
