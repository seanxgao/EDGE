"""
SQLite database layer using Peewee ORM.

Defines three tables:
- nodes_context: Immutable or rarely-changed content data
- neighbors: Graph connectivity (edges)
- nodes_usage: Frequently updated behavioral data
"""

import os
from datetime import datetime
from peewee import (
    SqliteDatabase,
    Model,
    IntegerField,
    TextField,
    TimestampField,
    FloatField,
    PrimaryKeyField,
    ForeignKeyField,
    CompositeKey,
)


# Database connection
_db_path = "memory.db"
_db = SqliteDatabase(_db_path)


class BaseModel(Model):
    """Base model for all tables."""

    class Meta:
        database = _db


class NodesContext(BaseModel):
    """Content layer: immutable or rarely-changed node data."""

    id = PrimaryKeyField()
    sentence_offset = IntegerField()  # index into sentences.jsonl
    embedding_index = IntegerField()  # index into embeddings.npy
    created_at = TimestampField(default=datetime.now)
    source = TextField(null=True)
    tag = TextField(null=True)
    language = TextField(null=True)
    initial_context = TextField(null=True)
    embedding_norm = FloatField(null=True)

    class Meta:
        table_name = "nodes_context"


class Neighbors(BaseModel):
    """Structural layer: graph connectivity (edges)."""

    u = IntegerField()  # from node id
    v = IntegerField()  # to node id
    weight = FloatField()
    edge_type = IntegerField()

    class Meta:
        table_name = "neighbors"
        primary_key = CompositeKey("u", "v")


class NodesUsage(BaseModel):
    """Usage layer: frequently updated behavioral data."""

    id = PrimaryKeyField()
    access_count = IntegerField(default=0)
    last_access_time = TimestampField(null=True)
    recent_hit_count = IntegerField(default=0)
    decay_score = FloatField(default=0.0)
    popularity = FloatField(default=0.0)

    class Meta:
        table_name = "nodes_usage"


def init_db(db_path="memory.db"):
    """
    Initialize database and create tables if they don't exist.

    Args:
        db_path: Path to SQLite database file
    """
    global _db, _db_path
    _db_path = db_path
    _db = SqliteDatabase(_db_path)

    # Update database for all models
    BaseModel._meta.database = _db
    NodesContext._meta.database = _db
    Neighbors._meta.database = _db
    NodesUsage._meta.database = _db

    # Create tables
    _db.connect()
    _db.create_tables([NodesContext, Neighbors, NodesUsage], safe=True)

    # Create index on neighbors.u for fast neighbor lookups
    try:
        _db.execute_sql("CREATE INDEX IF NOT EXISTS idx_neighbors_u ON neighbors(u);")
    except Exception:
        pass  # Index might already exist

    return _db


def add_node_context(
    node_id,
    sentence_offset,
    embedding_index,
    source=None,
    tag=None,
    language=None,
    initial_context=None,
    embedding_norm=None,
):
    """
    Add a new node context record.

    Args:
        node_id: Node identifier (INTEGER)
        sentence_offset: Index into sentences.jsonl
        embedding_index: Index into embeddings.npy
        source: Optional source text
        tag: Optional tag
        language: Optional language code
        initial_context: Optional initial context text
        embedding_norm: Optional embedding norm value

    Returns:
        NodesContext instance
    """
    node = NodesContext.create(
        id=node_id,
        sentence_offset=sentence_offset,
        embedding_index=embedding_index,
        source=source,
        tag=tag,
        language=language,
        initial_context=initial_context,
        embedding_norm=embedding_norm,
    )
    return node


def add_node_usage(
    node_id,
    access_count=0,
    last_access_time=None,
    recent_hit_count=0,
    decay_score=0.0,
    popularity=0.0,
):
    """
    Add a new node usage record.

    Args:
        node_id: Node identifier (INTEGER)
        access_count: Initial access count
        last_access_time: Initial last access time
        recent_hit_count: Initial recent hit count
        decay_score: Initial decay score
        popularity: Initial popularity score

    Returns:
        NodesUsage instance
    """
    usage = NodesUsage.create(
        id=node_id,
        access_count=access_count,
        last_access_time=last_access_time,
        recent_hit_count=recent_hit_count,
        decay_score=decay_score,
        popularity=popularity,
    )
    return usage


def add_neighbor(u, v, weight, edge_type):
    """
    Add a graph edge (neighbor relationship).

    Args:
        u: Source node id
        v: Target node id
        weight: Edge weight
        edge_type: Edge type code (INTEGER)

    Returns:
        Neighbors instance
    """
    neighbor, created = Neighbors.get_or_create(
        u=u, v=v, defaults={"weight": weight, "edge_type": edge_type}
    )
    if not created:
        neighbor.weight = weight
        neighbor.edge_type = edge_type
        neighbor.save()
    return neighbor


def get_neighbors(u):
    """
    Get all neighbors of node u.

    Args:
        u: Source node id

    Returns:
        List of dicts with keys: v, weight, edge_type
    """
    neighbors = Neighbors.select().where(Neighbors.u == u)
    return [
        {"v": n.v, "weight": n.weight, "edge_type": n.edge_type} for n in neighbors
    ]


def get_node_context(node_id):
    """
    Get node context by id.

    Args:
        node_id: Node identifier

    Returns:
        Dict with node context fields, or None if not found
    """
    try:
        node = NodesContext.get_by_id(node_id)
        return {
            "id": node.id,
            "sentence_offset": node.sentence_offset,
            "embedding_index": node.embedding_index,
            "created_at": node.created_at,
            "source": node.source,
            "tag": node.tag,
            "language": node.language,
            "initial_context": node.initial_context,
            "embedding_norm": node.embedding_norm,
        }
    except NodesContext.DoesNotExist:
        return None


def get_node_usage(node_id):
    """
    Get node usage by id.

    Args:
        node_id: Node identifier

    Returns:
        Dict with node usage fields, or None if not found
    """
    try:
        usage = NodesUsage.get_by_id(node_id)
        return {
            "id": usage.id,
            "access_count": usage.access_count,
            "last_access_time": usage.last_access_time,
            "recent_hit_count": usage.recent_hit_count,
            "decay_score": usage.decay_score,
            "popularity": usage.popularity,
        }
    except NodesUsage.DoesNotExist:
        return None


def get_node(node_id):
    """
    Get complete node information (context + usage + neighbors).

    Args:
        node_id: Node identifier

    Returns:
        Dict with keys: context, usage, neighbors
        Each key may be None if the corresponding data doesn't exist
    """
    context = get_node_context(node_id)
    usage = get_node_usage(node_id)
    neighbors = get_neighbors(node_id)

    return {
        "context": context,
        "usage": usage,
        "neighbors": neighbors,
    }


def update_node_usage(
    node_id,
    access_count=None,
    last_access_time=None,
    recent_hit_count=None,
    decay_score=None,
    popularity=None,
):
    """
    Update node usage fields.

    Args:
        node_id: Node identifier
        access_count: Optional new access count
        last_access_time: Optional new last access time
        recent_hit_count: Optional new recent hit count
        decay_score: Optional new decay score
        popularity: Optional new popularity score

    Returns:
        Updated NodesUsage instance, or None if not found
    """
    try:
        usage = NodesUsage.get_by_id(node_id)
        if access_count is not None:
            usage.access_count = access_count
        if last_access_time is not None:
            usage.last_access_time = last_access_time
        if recent_hit_count is not None:
            usage.recent_hit_count = recent_hit_count
        if decay_score is not None:
            usage.decay_score = decay_score
        if popularity is not None:
            usage.popularity = popularity
        usage.save()
        return usage
    except NodesUsage.DoesNotExist:
        return None

