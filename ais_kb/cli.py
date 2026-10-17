#!/usr/bin/env python3
"""Production CLI for ais-kb."""
import click
from pathlib import Path
from ais_kb.storage.db import ProductionKnowledgeStore
from ais_kb.ontology.grounder import ProductionGrounder
from ais_kb.ontology.disambiguator import AdeftDisambiguator

@click.group()
def cli():
    pass

@cli.command()
@click.option("--db", default=":memory:", help="DuckDB path")
@click.option("--paper-id", required=True)
def ingest(db, paper_id):
    store = ProductionKnowledgeStore(db_path=db)
    click.echo(f"Ingest pipeline ready for {paper_id}")

@cli.command()
@click.option("--db", default=":memory:")
@click.option("--min-belief", default=0.5)
def query(db, min_belief):
    store = ProductionKnowledgeStore(db_path=db)
    claims = store.query_claims(min_belief=min_belief)
    click.echo(f"Found {len(claims)} claims with belief >= {min_belief}")

if __name__ == "__main__":
    cli()
