import os
import json
import time
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel


class PsychologicalDepthAnalysis(BaseModel):
    """Analysis of psychological depth in responses."""
    before_level: Literal["powierzchowny", "średni", "głęboki"]
    after_level: Literal["powierzchowny", "średni", "głęboki"]
    before_focus: Literal["psychologiczny", "mieszany", "behawioralny/zewnętrzny"]
    after_focus: Literal["psychologiczny", "mieszany", "behawioralny/zewnętrzny"]
    change_direction: Literal["bardziej psychologiczny", "bardziej behawioralny", "bez zmian"]


class ReactionTypes(BaseModel):
    """Categorization of reaction types."""
    emocjonalne: int
    kognitywne: int
    behawioralne_manifestacje: int
    zewnętrzne_wyniki: int


class ReactionTypesAnalysis(BaseModel):
    """Analysis of reaction types before and after."""
    before_types: ReactionTypes
    after_types: ReactionTypes
    emotional_granularity_change: str


class ConceptualCoverage(BaseModel):
    """Coverage of conceptual categories."""
    before_categories: List[str]
    after_categories: List[str]
    maintained_categories: List[str]
    new_categories: List[str]
    lost_categories: List[str]


class EvolutionAssessment(BaseModel):
    """Assessment of how understanding evolved."""
    type: str
    description: str
    quality_change: float


class ReactionAnalysisResult(BaseModel):
    """Schema for reaction analysis result from OpenAI."""
    semantic_similarity: float
    psychological_depth_analysis: PsychologicalDepthAnalysis
    reaction_types_analysis: ReactionTypesAnalysis
    conceptual_coverage: ConceptualCoverage
    evolution_assessment: EvolutionAssessment
    key_insights: str


# Nested models for Intention Analysis
class ConceptualDepthAnalysis(BaseModel):
    """Analysis of conceptual depth in responses."""
    before_level: Literal["powierzchowny", "średni", "głęboki"]
    after_level: Literal["powierzchowny", "średni", "głęboki"]
    before_focus: Literal["koncepcyjny", "mieszany", "konkretny"]
    after_focus: Literal["koncepcyjny", "mieszany", "konkretny"]
    change_direction: Literal["bardziej koncepcyjny", "bardziej konkretny", "bez zmian"]


class IntentionAnalysisResult(BaseModel):
    """Schema for intention analysis result from OpenAI."""
    semantic_similarity: float
    conceptual_depth_analysis: ConceptualDepthAnalysis
    conceptual_coverage: ConceptualCoverage
    evolution_assessment: EvolutionAssessment
    key_insights: str


class ConsequencesDepthAnalysis(BaseModel):
    """Analysis of conceptual depth in consequences responses."""
    before_level: Literal["powierzchowny", "średni", "głęboki"]
    after_level: Literal["powierzchowny", "średni", "głęboki"]
    before_focus: Literal["konkretny", "mieszany", "koncepcyjny"]
    after_focus: Literal["konkretny", "mieszany", "koncepcyjny"]
    change_direction: Literal["bardziej koncepcyjny", "bardziej konkretny", "bez zmian"]
    before_horizon: Literal["wąski", "czasowy", "systemowy"]
    after_horizon: Literal["wąski", "czasowy", "systemowy"]
    horizon_change: Literal["poszerzenie horyzontu", "zawężenie horyzontu", "bez zmian"]


class ConsequenceCoverage(BaseModel):
    """Coverage of short-term and long-term consequences."""
    short_term_before: List[str]
    long_term_before: List[str]
    short_term_after: List[str]
    long_term_after: List[str]


class ConsequenceReach(BaseModel):
    """Reach levels of consequences."""
    before_levels: List[Literal["indywidualny", "relacyjny", "instytucjonalny", "społeczny"]]
    after_levels: List[Literal["indywidualny", "relacyjny", "instytucjonalny", "społeczny"]]
    new_levels: List[str]
    lost_levels: List[str]


class ConsequenceIntegration(BaseModel):
    """Integration of consequences across levels."""
    before: Literal["brak", "lista skutków", "powiązane poziomy"]
    after: Literal["brak", "lista skutków", "powiązane poziomy"]


class ConsequencesAnalysisResult(BaseModel):
    """Schema for consequences analysis result from OpenAI."""
    semantic_similarity: float
    conceptual_depth_analysis: ConsequencesDepthAnalysis
    consequence_coverage: ConsequenceCoverage
    consequence_reach: ConsequenceReach
    consequence_integration: ConsequenceIntegration
    evolution_assessment: EvolutionAssessment
    key_insights: str


RESPONSE_FORMATS = {
    'reaction': ReactionAnalysisResult,
    'intention': IntentionAnalysisResult,
    'consequences': ConsequencesAnalysisResult
}


class AnnotatorCompetenceAnalyzer:
    """Analyzes annotator competences using OpenAI structured outputs."""

    def __init__(
        self,
        openai_api_key: str,
        csv_path: str,
        max_workers: int = 2,
        request_delay: float = 1.0,
        max_retries: int = 5
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.csv_path = csv_path
        self.max_workers = max_workers
        self.request_delay = request_delay  # Delay between requests in seconds
        self.max_retries = max_retries
        self.template_dir = Path(__file__).parent
        self.last_request_time = 0  # Track last request time for rate limiting

        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        self.templates = {}
        for analysis_type in ['reaction', 'intention', 'consequences']:
            self.templates[analysis_type] = {
                'system': self.jinja_env.get_template(f'prompts/{analysis_type}_prompt.j2'),
                'user': self.jinja_env.get_template(f'prompts/user_prompt.j2')
            }

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    COLUMN_CONFIG = {
        'intention': {
            'main': ['first_question_4_intention', 'repeated_question_4_intention']
        },
        'reaction': {
            'main': ['first_question_9_reaction', 'repeated_question_9_reaction']
        },
        'consequences': {
            'main': ['first_question_5_consequences', 'repeated_question_5_consequences'],
            'extra': ['first_question_6_consequences_severity', 'repeated_question_6_consequences_severity']
        }
    }

    def _get_required_columns(self, analysis_types: List[str]) -> List[str]:
        """Get required columns for specified analysis types."""
        required_columns = []
        for analysis_type in analysis_types:
            config = self.COLUMN_CONFIG[analysis_type]
            required_columns.extend(config['main'])
            if 'extra' in config:
                required_columns.extend(config['extra'])
        return required_columns

    def prepare_data(self, df: pd.DataFrame, analysis_types: List[str]) -> pd.DataFrame:
        """Prepare data by selecting relevant columns for specified analysis types."""
        required_columns = self._get_required_columns(analysis_types)

        additional_columns = [
            'id',
            'example',
            'annotator',
            'first_question_10_certainty',
            'repeated_question_10_certainty',
            'annotator_group',
            'first_question_3_intention_clarity',
            'repeated_question_3_intention_clarity',
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Include additional columns if they exist
        all_columns = required_columns + [col for col in additional_columns if col in df.columns]
        return df[all_columns].copy()

    def _get_template_context(
        self,
        row: pd.Series,
        analysis_type: str
    ) -> dict:
        """Get template context variables for a given analysis type."""
        if analysis_type == 'reaction':
            return {
                'before': row['first_question_9_reaction'],
                'after': row['repeated_question_9_reaction']
            }
        elif analysis_type == 'intention':
            return {
                'before': row['first_question_4_intention'],
                'after': row['repeated_question_4_intention']
            }
        elif analysis_type == 'consequences':
            return {
                'before': row['first_question_5_consequences'],
                'after': row['repeated_question_5_consequences'],
                'before_severity': row['first_question_6_consequences_severity'],
                'after_severity': row['repeated_question_6_consequences_severity']
            }
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")

    def render_prompts(
        self,
        row: pd.Series,
        analysis_type: Literal["reaction", "intention", "consequences"]
    ) -> tuple[str, str]:
        templates = self.templates[analysis_type]
        context = self._get_template_context(row, analysis_type)

        system_prompt = templates['system'].render()
        user_prompt = templates['user'].render(**context)

        return system_prompt, user_prompt

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Recursively flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], str):
                    items.append((new_key, '; '.join(v)))
                else:
                    items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _rate_limited_api_call(self, system_prompt: str, user_prompt: str, response_format):
        for attempt in range(self.max_retries):
            try:
                # Rate limiting: ensure minimum delay between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    time.sleep(sleep_time)

                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=response_format,
                    temperature=0
                )

                self.last_request_time = time.time()
                return response

            except (RateLimitError, APIError) as e:
                # Exponential backoff for rate limit and API errors
                # Use longer wait time for rate limits
                wait_time = (2 ** attempt) * (2 if isinstance(e, RateLimitError) else 1)
                error_type = "Rate limit" if isinstance(e, RateLimitError) else "API error"
                print(f"  {error_type} (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)

                if attempt == self.max_retries - 1:
                    raise  # Re-raise on final attempt

    def analyze_single_row(
        self,
        row: pd.Series,
        index: int,
        analysis_type: Literal["reaction", "intention", "consequences"]
    ) -> dict:
        system_prompt, user_prompt = self.render_prompts(row, analysis_type)
        response_format = RESPONSE_FORMATS[analysis_type]

        try:
            response = self._rate_limited_api_call(system_prompt, user_prompt, response_format)
            result = response.choices[0].message.parsed

            # Convert the result to a flat dictionary
            result_dict = self._flatten_dict(result.model_dump())
            result_dict['index'] = index
            result_dict['analysis_type'] = analysis_type

            return result_dict

        except Exception as e:
            print(f"Error processing row {index} for {analysis_type}: {e}")
            return {
                'index': index,
                'analysis_type': analysis_type,
                'error': str(e)
            }

    def _get_checkpoint_path(self, analysis_type: str) -> Path:
        """Get the checkpoint file path for a given analysis type."""
        return Path(__file__).parent / f"checkpoint_{analysis_type}.json"

    def _load_checkpoint(self, analysis_type: str) -> dict:
        """Load checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path(analysis_type)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, analysis_type: str, results: List[dict]):
        """Save checkpoint to disk."""
        checkpoint_path = self._get_checkpoint_path(analysis_type)
        checkpoint_data = {str(r['index']): r for r in results}
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    def analyze_all_rows(
        self,
        df: pd.DataFrame,
        analysis_type: Literal["reaction", "intention", "consequences"],
        save_every: int = 5
    ) -> List[dict]:
        # Load existing checkpoint
        checkpoint = self._load_checkpoint(analysis_type)
        completed_indices = set(int(idx) for idx in checkpoint.keys())

        if completed_indices:
            print(f"Found checkpoint with {len(completed_indices)} completed rows. Resuming...")
            results = list(checkpoint.values())
        else:
            results = []

        # Filter out already completed rows
        pending_rows = [(idx, row) for idx, row in df.iterrows() if idx not in completed_indices]

        if not pending_rows:
            print(f"All rows already completed for {analysis_type} analysis!")
            return results

        print(f"Processing {len(pending_rows)} remaining rows...")
        completed_count = len(completed_indices)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.analyze_single_row, row, idx, analysis_type): idx
                for idx, row in pending_rows
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    print(f"Completed {analysis_type} analysis for row {idx + 1}/{len(df)} ({completed_count} total)")

                    if completed_count % save_every == 0:
                        self._save_checkpoint(analysis_type, results)
                        print(f"  → Checkpoint saved ({completed_count} rows)")

                except Exception as e:
                    print(f"Exception for row {idx} ({analysis_type}): {e}")
                    results.append({
                        'index': idx,
                        'analysis_type': analysis_type,
                        'error': str(e)
                    })
                    completed_count += 1

        self._save_checkpoint(analysis_type, results)
        print(f"Final checkpoint saved for {analysis_type}")

        return results

    def clear_checkpoints(self, analysis_types: List[str] = None):
        """Remove checkpoint files for specified analysis types."""
        if analysis_types is None:
            analysis_types = ["reaction", "intention", "consequences"]

        for analysis_type in analysis_types:
            checkpoint_path = self._get_checkpoint_path(analysis_type)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Removed checkpoint: {checkpoint_path}")

    def run(
        self,
        analysis_types: List[str],
        clear_checkpoints_flag: bool = False
    ) -> tuple[pd.DataFrame, dict]:
        """
        Main execution method.

        Args:
            analysis_types: List of analysis types to run ('intention', 'reaction', 'consequences')
            clear_checkpoints_flag: If True, remove existing checkpoints and start fresh

        Returns:
            Tuple of (final_df, results_dict) where results_dict maps analysis_type to DataFrame
        """
        df = self.load_data()
        print(f"Loaded {len(df)} rows")

        if clear_checkpoints_flag:
            print("\nClearing existing checkpoints...")
            self.clear_checkpoints(analysis_types)

        print("Preparing data...")
        prepared_df = self.prepare_data(df, analysis_types)

        results_dict = {}

        # Run each requested analysis
        for analysis_type in analysis_types:
            print(f"\nRunning {analysis_type} analysis for {len(prepared_df)} rows...")
            results = self.analyze_all_rows(prepared_df, analysis_type)
            results_df = pd.DataFrame(results)
            results_df = results_df.set_index('index')
            results_df = results_df.add_prefix(f'{analysis_type}_')
            results_dict[analysis_type] = results_df

        # Join all results with original data
        final_df = prepared_df
        for analysis_df in results_dict.values():
            final_df = final_df.join(analysis_df)

        print(f"\nCompleted! Final DataFrame has {len(final_df)} rows and {len(final_df.columns)} columns")

        # Clean up checkpoints after successful completion
        print("\nCleaning up checkpoints...")
        self.clear_checkpoints(analysis_types)

        return final_df, results_dict


def main():
    parser = argparse.ArgumentParser(
        description='Analyze annotator competences using OpenAI structured outputs'
    )
    parser.add_argument(
        '--analysis-types',
        nargs='+',
        choices=['intention', 'reaction', 'consequences'],
        default=['intention', 'reaction', 'consequences'],
        help='Types of analysis to run (default: intention reaction consequences)'
    )
    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear existing checkpoints and start fresh'
    )
    parser.add_argument(
        '--csv-path',
        default="data/AB_comparison.csv",
        help='Path to CSV file with data'
    )

    args = parser.parse_args()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    print(f"Running analysis types: {', '.join(args.analysis_types)}")

    analyzer = AnnotatorCompetenceAnalyzer(openai_api_key, args.csv_path)
    final_df, results_dict = analyzer.run(args.analysis_types, args.clear_checkpoints)

    output_dir = Path(__file__).parent
    final_output_path = output_dir / "annotated_results_full.csv"

    final_df.to_csv(final_output_path, index=False)
    print(f"\nResults saved to:")
    print(f"  - Full results: {final_output_path}")

    # Save individual analysis results
    for analysis_type, analysis_df in results_dict.items():
        output_path = output_dir / f"{analysis_type}_analysis.csv"
        analysis_df.to_csv(output_path)
        print(f"  - {analysis_type.capitalize()} analysis: {output_path}")

    return final_df


if __name__ == "__main__":
    result = main()
    print("\nFirst few rows of results:")
    print(result.head())
