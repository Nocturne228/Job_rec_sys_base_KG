"""Typed skill-graph adapters with a shared, provenance-aware contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .loader import GraphLoader
from .models import GraphEntities


class SkillGraphStore(Protocol):
    def competency_evidence(self, user_id: str, job_id: str) -> Dict[str, Any]: ...
    def healthcheck(self) -> bool: ...


@dataclass
class InMemorySkillGraph:
    """Deterministic adapter used by tests and the self-contained demo."""

    entities: GraphEntities

    def __post_init__(self) -> None:
        self.loader = GraphLoader(self.entities)

    def competency_evidence(self, user_id: str, job_id: str) -> Dict[str, Any]:
        return self.loader.get_recommended_learning_path(user_id, job_id)

    def healthcheck(self) -> bool:
        return True


class Neo4jSkillGraph:
    """Neo4j implementation of the typed skill-graph contract.

    The adapter intentionally accepts an injected driver so integration tests can
    exercise query behavior without hiding database access behind global state.
    """

    def __init__(
        self, uri: str, user: str, password: str, driver: Optional[Any] = None
    ):
        if driver is None:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver

    def close(self) -> None:
        self.driver.close()

    def healthcheck(self) -> bool:
        with self.driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
            return record is not None and record["ok"] == 1

    def competency_evidence(self, user_id: str, job_id: str) -> Dict[str, Any]:
        gap_query = """
        MATCH (u:User {id: $user_id}), (j:Job {id: $job_id})
        MATCH (j)-[req:REQUIRES]->(target:Skill)
        OPTIONAL MATCH (u)-[has:HAS_SKILL]->(target)
        WITH target, req, has
        WHERE has IS NULL OR coalesce(has.level_value, 0) < coalesce(req.level_value, 1)
        RETURN target.id AS skill_id, req.level AS required_level,
               coalesce(has.level, null) AS user_level
        ORDER BY skill_id
        """
        path_query = """
        MATCH (u:User {id: $user_id})-[:HAS_SKILL]->(known:Skill)
        MATCH (j:Job {id: $job_id})-[:REQUIRES]->(target:Skill)
        MATCH p=shortestPath((known)-[:PREREQUISITE_OF*1..3]->(target))
        RETURN [n IN nodes(p) | n.id] AS skills,
               [r IN relationships(p) | {
                   source_skill_id: startNode(r).id,
                   target_skill_id: endNode(r).id,
                   relation_type: type(r),
                   confidence: coalesce(r.confidence, 1.0),
                   source: coalesce(r.source, 'unknown')
               }] AS evidence
        LIMIT 5
        """
        with self.driver.session() as session:
            gaps = [
                dict(row)
                for row in session.run(gap_query, user_id=user_id, job_id=job_id)
            ]
            paths = [
                dict(row)
                for row in session.run(path_query, user_id=user_id, job_id=job_id)
            ]
        return {
            "skill_gap": {
                g["skill_id"]: (g["user_level"], g["required_level"]) for g in gaps
            },
            "paths": paths,
            "missing_skills": [g["skill_id"] for g in gaps],
            "gap_count": len(gaps),
        }
