# Utility Scripts

This directory contains auxiliary scripts for visualization and data management.

## Neo4j Visualization Scripts

### `show_in_neo4j.py`
Basic Neo4j graph visualization script.

**Usage**:
```bash
python scripts/show_in_neo4j.py
```

### `show_in_neo4j_with_tag.py`
Neo4j visualization with entity tags and categorization.

**Usage**:
```bash
python scripts/show_in_neo4j_with_tag.py
```

### `coloring_for_neo4j.py`
Custom coloring for Neo4j graph nodes based on entity types.

**Usage**:
```bash
python scripts/coloring_for_neo4j.py
```

## Data Conversion Scripts

### `graphml_to_excel.py`
Convert GraphML format to Excel for offline analysis.

**Usage**:
```bash
python scripts/graphml_to_excel.py --input graph.graphml --output output.xlsx
```

### `graphrag_excel_to_neo4j.py`
Import Excel data into Neo4j database.

**Usage**:
```bash
python scripts/graphrag_excel_to_neo4j.py --input data.xlsx --neo4j-uri bolt://localhost:7687
```

---

## Requirements

These scripts require:
- Neo4j database running locally or remotely
- `neo4j` Python driver (included in requirements.txt)
- Graph data exported from DOKE-RAG

## Notes

- These scripts are **optional utilities** for visualization and data export
- Core DOKE-RAG functionality does not depend on them
- They are provided for convenience in analyzing and inspecting knowledge graphs
