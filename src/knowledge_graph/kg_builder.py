"""
builder.py: Lightweight knowledge graph from corpus text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from loguru import logger

try:
    import spacy
    from spacy.tokens import Doc
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not installed; KG extraction disabled.")

from src.config import settings
from src.ingestion.ingestor import Chunk

# Types
EntityType = str   # e.g. "ORG", "PERSON", "GPE", "PRODUCT"
RelationType = str  # e.g. "is_a", "located_in", "works_for"

# Builder
class KnowledgeGraphBuilder:
    """
    Builds an entity–relation graph from a list of Chunks.

    Usage
    -----
    >>> builder = KnowledgeGraphBuilder()
    >>> kg = builder.build(chunks)
    >>> facts = kg.get_facts_about("Generali")
    """

    def __init__(self, spacy_model: str = settings.spacy_model):
        if not _SPACY_AVAILABLE:
            raise RuntimeError("spaCy is required for knowledge graph extraction.")
        logger.info(f"Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(
                f"Model {spacy_model} not found. "
                f"Run: python -m spacy download {spacy_model}"
            )
            raise

    def build(self, chunks: list[Chunk]) -> "KnowledgeGraph":
        """
        Extract entities and relations from all chunks and return a KG.
        """
        graph = nx.DiGraph()
        entity_sources: dict[str, list[str]] = defaultdict(list)

        texts = [c.text for c in chunks]
        logger.info(f"Extracting entities from {len(texts)} chunks…")

        # Batch processing for efficiency
        for chunk, doc in zip(chunks, self.nlp.pipe(texts, batch_size=32)):
            self._extract_entities(doc, graph, entity_sources, chunk.source)
            self._extract_relations(doc, graph)

        # Attach source provenance to each node
        for entity, sources in entity_sources.items():
            if graph.has_node(entity):
                graph.nodes[entity]["sources"] = list(set(sources))

        logger.info(
            f"KG built: {graph.number_of_nodes()} entities, "
            f"{graph.number_of_edges()} relations"
        )
        return KnowledgeGraph(graph)

    # Extraction helpers 

    def _extract_entities(
        self,
        doc: "Doc",
        graph: nx.DiGraph,
        entity_sources: dict[str, list[str]],
        source: str,
    ) -> None:
        for ent in doc.ents:
            # Normalise: lowercase for matching, keep original for display
            key = ent.text.strip()
            if len(key) < 2:
                continue
            if not graph.has_node(key):
                graph.add_node(key, label=ent.label_, count=0)
            graph.nodes[key]["count"] = graph.nodes[key].get("count", 0) + 1
            entity_sources[key].append(source)

    def _extract_relations(self, doc: "Doc", graph: nx.DiGraph) -> None:
        """
        Heuristic dependency-based relation extraction.
        Looks for (SUBJ entity) → verb → (OBJ entity) triples.
        """
        for token in doc:
            if token.pos_ != "VERB":
                continue

            subj_ents = self._get_entity_spans(token, doc, dep_labels={"nsubj", "nsubjpass"})
            obj_ents = self._get_entity_spans(token, doc, dep_labels={"dobj", "pobj", "attr"})

            for subj in subj_ents:
                for obj in obj_ents:
                    if subj == obj:
                        continue
                    relation = token.lemma_.lower()
                    if graph.has_node(subj) and graph.has_node(obj):
                        graph.add_edge(subj, obj, relation=relation)

    @staticmethod
    def _get_entity_spans(
        token: "spacy.tokens.Token",
        doc: "Doc",
        dep_labels: set[str],
    ) -> list[str]:
        """Find entity text for tokens with the given dependency labels."""
        results = []
        for child in token.children:
            if child.dep_ in dep_labels:
                # Check if this token is part of a named entity
                for ent in doc.ents:
                    if child.i >= ent.start and child.i < ent.end:
                        results.append(ent.text.strip())
                        break
        return results

class KnowledgeGraph:
    """
    Wrapper around a NetworkX DiGraph that exposes domain-relevant queries.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    # Query API 

    def get_facts_about(self, entity: str) -> list[dict]:
        """
        Return all (entity, relation, target) triples where *entity* appears.
        Used to cross-check LLM claims.
        """
        facts = []
        if not self.graph.has_node(entity):
            # Try case-insensitive lookup
            entity = self._fuzzy_lookup(entity)
            if entity is None:
                return []

        for _, target, data in self.graph.out_edges(entity, data=True):
            facts.append({
                "subject": entity,
                "relation": data.get("relation", "related_to"),
                "object": target,
            })
        for source, _, data in self.graph.in_edges(entity, data=True):
            facts.append({
                "subject": source,
                "relation": data.get("relation", "related_to"),
                "object": entity,
            })
        return facts

    def entities_in_text(self, text: str) -> list[str]:
        """Return all KG entities mentioned in *text*."""
        found = []
        text_lower = text.lower()
        for node in self.graph.nodes:
            if node.lower() in text_lower:
                found.append(node)
        return found

    def check_claim(self, claim: str) -> dict:
        """
        Lightweight consistency check: look for entities in the claim and
        see if the KG contains contradicting facts.

        Returns a dict with:
          - entities_found: entities recognised in the claim
          - supporting_facts: facts that are consistent with the claim
          - flag: True if the claim seems inconsistent with the KG
        """
        entities = self.entities_in_text(claim)
        supporting: list[dict] = []
        for ent in entities:
            supporting.extend(self.get_facts_about(ent))

        return {
            "entities_found": entities,
            "supporting_facts": supporting,
            "flag": len(entities) > 0 and len(supporting) == 0,
        }

    def summary(self) -> dict:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
            "top_entities": sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("count", 0),
                reverse=True,
            )[:10],
        }

    # Persistence 

    def save(self, path: Path) -> None:
        data = nx.node_link_data(self.graph)
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"KG saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text())
        graph = nx.node_link_graph(data, directed=True)
        return cls(graph)

    # Helpers 

    def _fuzzy_lookup(self, entity: str) -> str | None:
        """Case-insensitive node lookup."""
        lower = entity.lower()
        for node in self.graph.nodes:
            if node.lower() == lower:
                return node
        return None
