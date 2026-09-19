"""EXCERPT on document access control in RAG. Not a standalone module.

Comments and docstrings translated to English for review; logic unchanged.

Assembled from four files of the production system (line numbers are the
originals'):
    routes/rag.py:30-56      _require_rag_write, _require_admin: the HTTP-layer
                             gates, RAG_ENABLED (404), then the permission (403);
    routes/rag.py:87-106     create_document: validation of the upload target,
                             the collection exists and a non-admin may only
                             upload into one they can reach;
    rag_collections.py:39-57 collection_exists, allowed_collection_ids: the
                             access resolution (public + user_collection grants);
    rag_search.py:65-151     rag_search: the same resolution as a SECOND copy of
                             the SQL, and the retrieval filter on collection_id;
    rag_ingest.py:266-272,   ingest_document: where collection_id comes from at
    rag_ingest.py:288-323    indexing time and how it is denormalised onto chunks.

The check is NOT centralised. That is a fact about the system, not a
simplification of the excerpt. What documents a user can see is decided in four
places: the rag_read permission at the tool entry (db.py, tool_rag_search, shown
in db_excerpt.py and not repeated here); the rag_write / admin permission at the
HTTP layer; the "public + grants" resolution in
rag_collections.allowed_collection_ids, used to validate the upload target; and
the same resolution written as a SEPARATE SQL statement inside rag_search, used
as the retrieval filter. The last two are independent queries with identical
meaning: they would drift apart silently, and there is no test that would catch
the drift. One resolution function serving both paths is the first thing I would
change here.

The retrieval filter sits on the denormalised rag_chunks.collection_id column:
its value comes from the RETURNING clause of the document insert, so the search
needs no join. The price of that decision is that moving a document to another
collection requires updating the chunks separately; there is no move code in the
system.

What is omitted: the remaining RAG routes (document listing, deletion, collection
and grant CRUD), the CRUD and grants in rag_collections, text extraction,
chunking, embedding and size limits in rag_ingest, and the retrieval settings
readers in rag_search.

Dependencies that are absent here: db, embeddings, rag_collections, rag_ingest,
auth, fastapi. The logic is untouched.

User-facing strings stay in Russian, as in production; their English meaning is
given in an adjacent comment.
"""

# --- routes/rag.py:30-56 - HTTP-layer gates -----------------------------------

def _require_rag_write(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Double gate: RAG_ENABLED first (404), then the rag_write permission (403)."""
    if not rag_ingest.is_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if not (current_user.get("permissions") or {}).get("rag_write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


def _require_admin(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Gate for collection/grant mutations: RAG_ENABLED (404), then admin (403).
    Kept apart from _require_rag_write: management is admin only (fail-closed)."""
    if not rag_ingest.is_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if not (current_user.get("permissions") or {}).get("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user


# --- routes/rag.py:87-106 - validation of the upload target -------------------

@router.post("/documents", status_code=status.HTTP_201_CREATED)
def create_document(
    file: UploadFile = File(...),
    collection_id: int = Form(...),
    current_user: dict[str, Any] = Depends(_require_rag_write),
) -> dict[str, Any]:
    # Target validation: the collection must exist; a non-admin may only upload
    # into one they can reach (public + their grants), an admin into any. Same
    # resolution as in rag_search.
    if not rag_collections.collection_exists(collection_id):
        raise HTTPException(
            # "Collection not found"
            status_code=status.HTTP_400_BAD_REQUEST, detail="Коллекция не найдена"
        )
    is_admin = bool((current_user.get("permissions") or {}).get("admin"))
    if not is_admin and collection_id not in rag_collections.allowed_collection_ids(
        int(current_user["id"])
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            # "No access to the specified collection"
            detail="Нет доступа к указанной коллекции",
        )


# --- rag_collections.py:39-57 - access resolution (copy no. 1) ----------------

def collection_exists(collection_id: int) -> bool:
    return bool(
        db._query("SELECT 1 FROM analyst.rag_collections WHERE id = %s", (collection_id,))
    )


def allowed_collection_ids(user_id: int) -> set[int]:
    """Collections a user can reach: public + user_collection grants. The same
    resolution as in rag_search, used to validate the upload target (non-admin)."""
    rows = db._query(
        """
        SELECT id FROM analyst.rag_collections WHERE is_public = true
        UNION
        SELECT collection_id FROM analyst.user_collection WHERE user_id = %s
        """,
        (user_id,),
    )
    return {int(r["id"]) for r in rows}


# --- rag_search.py:65-151 - resolution (copy no. 2) and the retrieval filter --

def rag_search(
    query: str,
    user_permissions: dict[str, Any] | None = None,
    user_id: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Find the chunks relevant to a query. The runtime half of the double gate:
    without RAG_ENABLED we do not go anywhere (fail-closed). Access is limited to
    the user's collections (public + user_collection grants). user_permissions has
    no effect on the filtering; the rag_read gate sits above this."""
    if not is_enabled():
        # "Document search is switched off."
        return {"error": "Поиск по документам отключён."}
    if not isinstance(query, str) or not query.strip():
        return {"type": "internal_document_context", "results": [], "message": EMPTY_MESSAGE}

    k = top_k if isinstance(top_k, int) and top_k > 0 else _top_k()
    query_vector = np.asarray(embeddings.embed_query(query), dtype=np.float32)

    conn = psycopg2.connect(**db._conn_params())
    try:
        register_vector(conn)
        # Access is resolved on entry: public collections + the user's grants.
        # is_public resolves into a set of ids rather than through grants.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM analyst.rag_collections WHERE is_public = true
                UNION
                SELECT collection_id FROM analyst.user_collection WHERE user_id = %s
                """,
                (user_id,),
            )
            allowed_collections = [row[0] for row in cur.fetchall()]

        # The filter uses the denormalised column on chunks (no join). An empty
        # list becomes ANY('{}'); the ::bigint[] cast gives the empty array a type.
        # The JOIN is there only for filename.
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.content,
                       d.filename AS source_document,
                       c.embedding <=> %s AS distance
                FROM analyst.rag_chunks c
                JOIN analyst.rag_documents d ON d.id = c.document_id
                WHERE c.collection_id = ANY(%s::bigint[])
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (query_vector, allowed_collections, query_vector, k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    min_score = _min_score()
    max_chars = _context_max_chars()
    results: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        distance = float(row["distance"])
        # The threshold is applied to score (higher = more relevant), while the
        # output carries distance.
        if min_score is not None and (1.0 - distance) < min_score:
            continue
        content = row["content"] or ""
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining]
        results.append({
            "content": content,
            "source_document": row["source_document"],
            "distance": round(distance, 6),
        })
        total += len(content)

    if not results:
        return {"type": "internal_document_context", "results": [], "message": EMPTY_MESSAGE}

    return {
        "type": "internal_document_context",
        "trusted": True,
        "note": NOTE,
        "query": query,
        "results": results,
        "result_count": len(results),
    }


# --- rag_ingest.py:266-272, 288-323 - where collection_id comes from ----------

def ingest_document(
    data: bytes,
    filename: str,
    uploaded_by: int,
    collection_id: int,
    content_type: str | None = None,
) -> int:
    # ... omitted: text extraction, chunking, embedding, sha256 ...
    conn = psycopg2.connect(**db._conn_params())
    try:
        register_vector(conn)  # adapts numpy vectors into the vector type
        with conn:  # commit on success, rollback on exception
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analyst.rag_documents
                        (filename, uploaded_by, collection_id, content_type,
                         size_bytes, sha256)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, collection_id
                    """,
                    (filename, uploaded_by, collection_id, resolved_type,
                     len(data), sha256),
                )
                doc_row = cur.fetchone()
                document_id = int(doc_row[0])
                collection_id = doc_row[1]  # confirmed from RETURNING for the chunks
                # collection_id is denormalised onto chunks: retrieval filters
                # without a join.
                execute_values(
                    cur,
                    """
                    INSERT INTO analyst.rag_chunks
                        (document_id, chunk_index, content, embedding, collection_id)
                    VALUES %s
                    """,
                    [
                        (document_id, index, chunk,
                         np.asarray(vector, dtype=np.float32), collection_id)
                        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                    ],
                )
        return document_id
    finally:
        conn.close()
