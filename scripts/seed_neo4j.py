#!/usr/bin/env python3
"""Idempotently load the deterministic demo graph into Neo4j."""

from __future__ import annotations

import os

from neo4j import GraphDatabase

from src.data import generate_mock_data


def seed():
    data = generate_mock_data(seed=42)
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session() as session:
        for skill in data.skills:
            session.run(
                "MERGE (s:Skill {id:$id}) SET s.name=$name, s.category=$category",
                id=skill.id,
                name=skill.name,
                category=skill.category,
            )
        for relation in data.skill_relations:
            session.run(
                """
                MATCH (a:Skill {id:$source}), (b:Skill {id:$target})
                MERGE (a)-[r:PREREQUISITE_OF]->(b)
                SET r.confidence=$confidence, r.source=$provenance
            """,
                source=relation.source_skill_id,
                target=relation.target_skill_id,
                confidence=relation.confidence,
                provenance=relation.source,
            )
        for user in data.users:
            session.run("MERGE (:User {id:$id})", id=user.id)
            for skill, level in user.skills.items():
                session.run(
                    """
                    MATCH (u:User {id:$user}), (s:Skill {id:$skill})
                    MERGE (u)-[r:HAS_SKILL]->(s)
                    SET r.level=$level, r.level_value=$value
                """,
                    user=user.id,
                    skill=skill,
                    level=level.value,
                    value=["beginner", "intermediate", "advanced", "expert"].index(
                        level.value
                    )
                    + 1,
                )
        for job in data.jobs:
            session.run("MERGE (:Job {id:$id})", id=job.id)
            for skill, level in job.required_skills.items():
                session.run(
                    """
                    MATCH (j:Job {id:$job}), (s:Skill {id:$skill})
                    MERGE (j)-[r:REQUIRES]->(s)
                    SET r.level=$level, r.level_value=$value
                """,
                    job=job.id,
                    skill=skill,
                    level=level.value,
                    value=["beginner", "intermediate", "advanced", "expert"].index(
                        level.value
                    )
                    + 1,
                )
    driver.close()


if __name__ == "__main__":
    seed()
    print("Neo4j demo graph seeded")
