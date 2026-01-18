# Evaluation Framework

This directory contains comprehensive evaluation scripts for DOKE-RAG.

## Directory Structure

```
evaluation/
├── sota_comparison.py       # Main SOTA comparison (DOKE-RAG vs LightRAG vs GraphRAG)
├── championship_eval.py      # Best configuration evaluation
├── token_usage.py            # Token consumption statistics
├── randomization/            # Answer randomization scripts
├── utils/                    # Utility scripts (data aggregation, analysis)
├── visualization/            # Plotting and visualization
└── archive/                  # Deprecated/old versions (preserved for reference)
```

## Core Evaluation Scripts

### 1. SOTA Comparison (`sota_comparison.py`)
**Purpose**: Compare DOKE-RAG against state-of-the-art baselines (LightRAG, GraphRAG)

**Features**:
- Pairwise comparison over 5 independent runs
- Multi-dimensional evaluation (Comprehensiveness, Diversity, Empowerment, Overall)
- Outputs detailed CSV reports and statistical summaries

**Usage**:
```bash
python evaluation/sota_comparison.py
```

**Output**: `Final_PK_Comparison_Report_5Runs/`

---

### 2. Championship Evaluation (`championship_eval.py`)
**Purpose**: Identify the best configuration across parameter grid search

**Features**:
- Reads consolidated Excel report from parameter grid experiments
- Runs round-robin tournament between "local champions"
- Identifies globally optimal configuration

**Usage**:
```bash
python evaluation/championship_eval.py
```

**Output**: `Championship_Evaluation_Final/`

---

### 3. Token Usage (`token_usage.py`)
**Purpose**: Track and compare token consumption across systems

**Features**:
- Compares DOKE-RAG, LightRAG, and GraphRAG
- Handles both flat and nested token structures
- Generates comparison reports

**Usage**:
```bash
python evaluation/token_usage.py
```

---

## Randomization Scripts

These scripts randomize answer order to eliminate positional bias in evaluation.

### Files:
- `randomization/single_randomize.py` - Single-pass randomization
- `randomization/batch_randomize.py` - Batch randomization
- `randomization/loop_randomize_dsr1.py` - Loop-based randomization (DSR1 dataset)
- `randomization/loop_randomize_dsv3.py` - Loop-based randomization (DSV3 dataset)

---

## Utility Scripts

### Data Consolidation (`utils/consolidate_results.py`)
**Purpose**: Merge multiple experiment CSV files into single Excel report

**Features**:
- Scans directories for `final_statistical_summary.csv`
- Aggregates results into Excel workbook
- Maintains experiment groupings

**Usage**:
```bash
python evaluation/utils/consolidate_results.py
```

**Output**: `Consolidated_Report.xlsx`

### Final Analysis (`utils/final_analysis.py`)
**Purpose**: Global statistical analysis of championship results

**Features**:
- Counts championship wins per configuration
- Calculates average rankings
- Generates final overall analysis

**Usage**:
```bash
python evaluation/utils/final_analysis.py
```

**Output**: `Final_Overall_Analysis.csv`

---

## Visualization Scripts

### Plotting (`visualization/plot_championship.py`)
**Purpose**: Visualize championship evaluation results

**Features**:
- Generates charts from championship data
- Highlights top-performing configurations

---

## Archive

The `archive/` directory contains deprecated or older versions of evaluation scripts, preserved for reference:
- `evaluate.py` (v1.0)
- `evaluate_v1.1.py`
- `evaluate_1_2.py`
- `evaluate_v1_zh_PK_en.py`

These scripts are **not recommended for use** but are kept for historical reference.

---

## Evaluation Workflow

### Typical Evaluation Pipeline:

```
1. Run Experiments (in experiments/)
   ↓
2. Randomize Answers (randomization/)
   ↓
3. SOTA Comparison (sota_comparison.py)
   ↓
4. Parameter Grid Search
   ↓
5. Consolidate Results (utils/consolidate_results.py)
   ↓
6. Championship Eval (championship_eval.py)
   ↓
7. Final Analysis (utils/final_analysis.py)
   ↓
8. Visualize Results (visualization/)
```

---

## Configuration

Most scripts require a `.env` file with API keys:
```
ALIYUN_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

See `../.env.example` for template.

---

## Output Format

### CSV Files
- `final_statistical_summary.csv` - Per-experiment statistics
- `Championship_Rankings_Final.csv` - Championship rankings
- `Final_Overall_Analysis.csv` - Global analysis

### Excel Files
- `Consolidated_Report.xlsx` - Aggregated experiment results

---

## Notes

- All evaluation uses DeepSeek-r1 as the judge model
- Each comparison is run 5 times to ensure statistical robustness
- Positional bias is eliminated through answer randomization
- Results are aggregated across multiple dimensions for comprehensive assessment
