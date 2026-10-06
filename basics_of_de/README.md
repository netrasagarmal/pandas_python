# Data Engineering Interview Notes

# PART 1: FOUNDATIONS

## 1. What is Big Data?

**What:** Data too large, too fast, or too varied for a single machine or traditional database to store and process efficiently.

**The Vs:**
| V | Meaning | Example |
|---|---|---|
| Volume | Terabytes to petabytes | Meta stores petabytes of photos |
| Velocity | Speed of arrival | 10,000 card swipes per second |
| Variety | Structured, semi-structured, unstructured | SQL tables, JSON logs, images, audio |
| Veracity | Trustworthiness and quality | Noisy IoT sensor data |
| Value | Business insight extracted | Recommendations, fraud detection |

**Origin:** Analyst Doug Laney defined the "3 Vs" in 2001. Google's papers on GFS (2003) and MapReduce (2004) inspired **Hadoop** (Doug Cutting and Mike Cafarella, 2006, at Yahoo), which made big data processing accessible on cheap commodity hardware.

**Why it matters:** One server hits limits on disk, RAM and CPU. The solution is to **scale out** (many machines) rather than **scale up** (a bigger machine).

**Example:** Uber processes millions of trips, GPS pings and payments every day, which is far beyond a single MySQL box.

> 🎤 *"Big data is when scale, speed or variety forces you to go distributed. The core idea is to split data across machines and bring the computation to the data."*

---

## 2. Batch Processing vs Stream Processing

| | Batch | Stream |
|---|---|---|
| Data | Bounded (finite chunk) | Unbounded (continuous) |
| Latency | Minutes to hours | Milliseconds to seconds |
| Trigger | Schedule (nightly) | Each event or micro-batch |
| Throughput | Very high | Lower per event |
| Complexity | Simpler | Harder (ordering, late data, state) |
| Tools | Spark, Hadoop, dbt, Airflow | Kafka, Flink, Spark Structured Streaming |
| Example | Nightly sales report, payroll | Fraud detection, live dashboards |

**Origin:** Batch is the oldest model, from the mainframe era (1950s–60s). Stream processing grew in the 2010s: Storm (Nathan Marz, 2011), Kafka (2011), Flink (2014), Spark Streaming (2013).

**Architectures:**
- **Lambda** (Nathan Marz): a batch layer plus a speed layer. It is accurate, but you maintain two codebases.
- **Kappa** (Jay Kreps, 2014): everything is a stream, and you replay the log for reprocessing. It is simpler.

**Key stream concepts to mention:** event time vs processing time, windowing (tumbling, sliding, session), watermarks for late data, exactly-once vs at-least-once delivery.

**Example:** A bank runs a batch job overnight for monthly statements, and a stream job to flag a suspicious transaction within 200 ms.

> 🎤 *"I choose batch when latency tolerance is high and cost efficiency matters. I choose streaming when the value of data decays quickly, such as fraud or ride dispatch."*

---

## 3. ETL vs ELT

| | ETL | ELT |
|---|---|---|
| Order | Extract → **Transform** → Load | Extract → Load → **Transform** |
| Transform happens | In a separate engine/staging server | Inside the warehouse/lake |
| Best for | On-prem, strict compliance, small curated data | Cloud, large scale, flexible analytics |
| Raw data kept? | Usually not | Yes |
| Schema | Schema-on-write | Schema-on-read/late |
| Tools | Informatica, Talend, SSIS | Fivetran/Airbyte + dbt + Snowflake/BigQuery |
| Cost model | Pay for transform servers | Pay for warehouse compute |

**Origin:** ETL came with data warehousing in the 1970s–90s (Informatica 1993). ELT rose when cloud warehouses such as BigQuery (2011), Redshift (2012) and Snowflake (2014) made compute cheap and elastic. **dbt** (2016) popularized SQL-based in-warehouse transformation.

**Example:**
- *ETL:* Mask PII in an on-prem server, then load clean data into Oracle DW.
- *ELT:* Fivetran dumps raw Salesforce data into Snowflake, and dbt models build `dim_customer` and `fact_orders`.

> 🎤 *"ELT preserves raw data, so you can reprocess when business logic changes. ETL is still right when you must filter or mask sensitive data before it lands."*

---

# PART 2: DATABASES

## 4. DBMS (Relational) vs NoSQL vs Graph vs Vector

| Type | Data model | Strengths | Weaknesses | Examples | Typical use |
|---|---|---|---|---|---|
| **RDBMS** | Tables, rows, SQL, joins | ACID, integrity, mature | Hard to scale horizontally | PostgreSQL, MySQL, Oracle | Banking, orders, ERP |
| **NoSQL: Key-Value** | key → value | Ultra-fast | No complex queries | Redis, DynamoDB | Sessions, cache |
| **NoSQL: Document** | JSON documents | Flexible schema | Joins are weak | MongoDB, Couchbase | Catalogs, CMS, profiles |
| **NoSQL: Wide-column** | Row key → column families | Massive writes, scale | Query-pattern-driven design | Cassandra, HBase, Bigtable | IoT, time-series, messaging |
| **Graph** | Nodes + edges + properties | Relationship traversal | Not for bulk analytics | Neo4j, Amazon Neptune | Fraud rings, social, knowledge graphs |
| **Vector** | High-dimensional embeddings | Similarity search (ANN) | Approximate, not for exact queries | Pinecone, Milvus, Weaviate, Qdrant, pgvector | RAG, semantic search, recommendations |

**Origins:**
- **RDBMS:** Edgar F. Codd (IBM) proposed the relational model in **1970**. SQL came from IBM (Chamberlin and Boyce, 1974). Oracle (1979), MySQL (1995) and PostgreSQL (1996) followed.
- **NoSQL:** Google **Bigtable** (2006) and Amazon **Dynamo** (2007) papers triggered it. Cassandra (2008), MongoDB (2009) and Redis (2009) came next. It was built for web-scale, availability and flexible schemas.
- **Graph:** Rooted in graph theory (Euler). Neo4j was released in 2007.
- **Vector:** Rose with deep learning embeddings, Meta's FAISS library (2017) and the HNSW algorithm (2016). Exploded in popularity with LLMs and RAG (2022 onwards).

**Examples:**
- *RDBMS:* `SELECT * FROM orders JOIN customers ...` for an e-commerce checkout.
- *Document:* A product with varying attributes (a shirt has size, a laptop has RAM) stored as JSON.
- *Wide-column:* Netflix stores viewing history in Cassandra.
- *Graph:* "Find all accounts within 3 hops of a known fraudster."
- *Vector:* Convert a question to an embedding, find the top-5 most similar document chunks, and feed them to an LLM.

**AI-engineer angle (this will impress):** Vector search uses **cosine similarity or dot product** with **ANN indexes** (HNSW, IVF) that trade a little accuracy for big speed. Many teams start with **pgvector** (Postgres extension) before adopting a dedicated vector DB.

**CAP theorem** (Eric Brewer, 2000): during a network partition you choose between Consistency and Availability. RDBMS is typically CP-leaning, while Cassandra and Dynamo are AP-leaning.

> 🎤 *"I pick the database by access pattern: transactions → RDBMS, flexible/high-write → NoSQL, relationships → graph, semantic similarity → vector."*

---

## 5. OLTP vs OLAP

| | OLTP | OLAP |
|---|---|---|
| Full form | Online Transaction Processing | Online Analytical Processing |
| Purpose | Run the business (day-to-day ops) | Analyze the business (decisions) |
| Queries | Short, simple, point lookups, INSERT/UPDATE | Complex aggregations, scans, joins |
| Data size per query | A few rows | Millions to billions of rows |
| Storage | Row-oriented | Column-oriented |
| Schema | Normalized (3NF) | Denormalized (star/snowflake) |
| Users | Customers, apps (thousands) | Analysts, BI tools (fewer) |
| Latency | Milliseconds | Seconds to minutes |
| Data | Current | Historical |
| Examples | PostgreSQL, MySQL, Oracle | Snowflake, BigQuery, Redshift, ClickHouse |
| Use case | ATM withdrawal, placing an order | "Revenue by region by quarter" |

**Origin:** OLTP dates to the 1960s (airline reservation systems such as SABRE) and 1970s banking. The term **OLAP** was coined by **Edgar Codd in 1993**.

**OLAP concepts:** cubes, dimensions (time, region, product), measures (sales), and operations such as slice, dice, drill-down, roll-up and pivot.

**Flow:** `OLTP DB → (ETL/ELT/CDC) → Warehouse (OLAP) → BI dashboards`

**Also know:** HTAP (hybrid, e.g., TiDB, SingleStore) combines both.

> 🎤 *"You never run heavy analytics on the production OLTP database, because it would slow down customers. You replicate to an OLAP system."*

---

# PART 3: ANALYTICAL STORAGE ARCHITECTURES

## 6. Data Warehouse vs Data Lake (vs Lakehouse)

| | Data Warehouse | Data Lake | Lakehouse |
|---|---|---|---|
| Data | Structured, curated | Raw: structured, semi-structured, unstructured | All |
| Schema | Schema-on-write | Schema-on-read | Both |
| Users | Analysts, BI | Data scientists, ML engineers | Everyone |
| Cost | Higher | Cheap (object storage) | Cheap |
| ACID/governance | Strong | Weak by default | Strong (via table formats) |
| Examples | Snowflake, BigQuery, Redshift, Teradata | S3, ADLS, GCS, HDFS | Databricks + Delta Lake, Apache Iceberg, Apache Hudi |

**Origin:**
- **Data warehouse:** Bill Inmon (the "father of data warehousing", 1990s, top-down approach) and Ralph Kimball (1996, dimensional modeling with star schemas, bottom-up).
- **Data lake:** Coined by **James Dixon** (CTO, Pentaho) in **2010**. His analogy was that a datamart is bottled water (cleaned, packaged) and a lake is the natural body of water.
- **Lakehouse:** Databricks (around 2020) combined lake storage with warehouse-grade features. Table formats: Hudi (Uber), Iceberg (Netflix, 2018), Delta Lake (Databricks, 2019).

**Problem with lakes alone:** the "data swamp" (no quality, no schema, no ACID). Lakehouses fix this.

**Examples:**
- *Warehouse:* A CFO dashboard on clean `fact_sales` in Snowflake.
- *Lake:* Store raw clickstream JSON, images and PDFs in S3 for ML training.

> 🎤 *"Warehouse for trusted BI, lake for cheap raw and ML data, and lakehouse tries to give one copy of data with both capabilities."*

---

## 7. Medallion Architecture

**What:** A layered data design pattern that progressively improves data quality as it moves through three tiers.

```
Sources → [ BRONZE ] → [ SILVER ] → [ GOLD ] → BI / ML / AI / APIs
            raw         cleaned      business-ready
```

| Layer | Content | Operations | Consumer |
|---|---|---|---|
| 🥉 **Bronze** | Raw, as-is, append-only copy of source | Ingest only, add metadata (load time, source) | Data engineers, replay/audit |
| 🥈 **Silver** | Cleaned, deduplicated, validated, conformed, joined | Type casting, null handling, dedup, SCD, schema enforcement | Data scientists, analysts |
| 🥇 **Gold** | Aggregated, business-level, modeled (star schema, KPIs, feature tables) | Aggregations, business logic | BI dashboards, ML models, executives |

**Origin:** Popularized by **Databricks** as the recommended design on Delta Lake. It is also called **multi-hop architecture**.

**Why use it:**
- Raw data is preserved, so you can reprocess after a bug or logic change.
- Quality improves step by step, and each layer has a clear contract.
- Debugging and lineage are easier.
- Different consumers use different layers.

**Example (e-commerce):**
- **Bronze:** `orders_raw`, JSON from Kafka with duplicates and bad dates.
- **Silver:** `orders_clean`, deduplicated, dates fixed, and joined with `customers`.
- **Gold:** `daily_revenue_by_region`, ready for the dashboard.

**AI angle:** Bronze holds raw documents, Silver holds cleaned and chunked text, and Gold holds embeddings and feature tables.

> 🎤 *"Bronze is the source of truth, Silver is the trusted data, and Gold is the data product."*

---

# PART 4: DISTRIBUTED PROCESSING ENGINES

## 8. MapReduce

**What:** A programming model for processing huge datasets in parallel across a cluster.

**Phases:**
1. **Map:** Each node processes its data split and emits `(key, value)` pairs.
2. **Shuffle & Sort:** Pairs with the same key are grouped and sent to the same reducer.
3. **Reduce:** Aggregates the values per key.

**Word-count example:**
```
Input:   "cat dog cat"
Map:     (cat,1) (dog,1) (cat,1)
Shuffle: cat → [1,1]   dog → [1]
Reduce:  (cat,2) (dog,1)
```

**Origin:** Google paper by **Jeffrey Dean and Sanjay Ghemawat (2004)**. It was implemented open-source in **Hadoop** (with HDFS storage) in 2006.

**Why:**
- Runs on cheap commodity machines.
- **Fault tolerance:** failed tasks are re-run on another node.
- **Data locality:** code moves to the data, not data to the code.
- Hides the complexity of parallelism from the developer.

**Limitations:** It writes intermediate results to disk, so it is slow. It is poor for iterative work (ML) and verbose to code. This is why Spark replaced it.

---

## 9. Apache Spark

**What:** A unified, in-memory distributed compute engine for batch, streaming, SQL, ML and graph processing.

**Origin:** **Matei Zaharia**, UC Berkeley AMPLab, **2009**. It was open-sourced in 2010 and became an Apache project in 2013. The creators founded **Databricks** (2013).

**Why Spark beats MapReduce:** It keeps intermediate data in **memory** (up to ~10–100x faster for some workloads). It has a rich API, supports iterative algorithms, and offers one engine for many workloads.

**Core concepts:**
| Concept | Meaning |
|---|---|
| Driver / Executors | Driver plans the job and executors run the tasks on worker nodes |
| RDD → DataFrame → Dataset | Low-level to high-level, optimized APIs |
| Lazy evaluation | Transformations build a plan, and **actions** trigger execution |
| DAG | Directed acyclic graph of stages |
| Narrow vs wide transformations | Narrow (`filter`, `map`) need no shuffle. Wide (`groupBy`, `join`) cause a **shuffle** (expensive) |
| Catalyst optimizer | Optimizes query plans |
| Components | Spark SQL, Structured Streaming, MLlib, GraphX |

**Example (PySpark):**
```python
df = spark.read.parquet("s3://bucket/orders/")
result = (df.filter("status = 'PAID'")
            .groupBy("region")
            .sum("amount"))
result.write.mode("overwrite").saveAsTable("gold.revenue_by_region")
```

**Performance tips to mention:** partition tuning, avoiding skew, broadcast joins for small tables, caching reused DataFrames, using Parquet or Delta.

> 🎤 *"MapReduce proved distributed processing was possible, and Spark made it fast and developer-friendly by doing it in memory with a DAG optimizer."*

---

# PART 5: SPEED AND MESSAGING

## 10. Caching and Redis

**What:** Storing frequently accessed data in fast memory so you avoid repeated slow work (database queries, API calls, computation).

**Strategies:**
| Pattern | How it works |
|---|---|
| **Cache-aside (lazy)** | App checks cache, on a miss reads the DB, then fills the cache. Most common |
| Read-through | Cache layer loads from the DB itself |
| Write-through | Writes go to the cache and DB together (consistent, slower) |
| Write-back (behind) | Writes go to the cache first, then to the DB asynchronously (fast, risk of loss) |

**Eviction:** LRU, LFU, TTL (time to live). **Challenges:** stale data, cache invalidation, cache stampede.

**Redis (Remote Dictionary Server):**
- An in-memory data store created by **Salvatore Sanfilippo** in **2009**. Earlier alternative: Memcached (2003, Brad Fitzpatrick).
- Data structures: strings, hashes, lists, sets, sorted sets, streams, bitmaps, HyperLogLog, geospatial.
- Optional persistence (RDB snapshots, AOF log), replication, clustering and Pub/Sub.
- Single-threaded command execution gives very low latency (sub-millisecond).

**Use cases and examples:**
- Session store (login sessions)
- Rate limiting (`INCR` with expiry)
- Leaderboards (sorted sets)
- Caching hot product pages
- Distributed locks, job queues
- **AI use:** **semantic cache** for LLM responses (skip repeated LLM calls), and vector search (Redis Stack)

```python
val = redis.get("user:42")
if not val:
    val = db.query(...)
    redis.setex("user:42", 3600, val)  # TTL 1 hour
```

> 🎤 *"A cache trades memory and freshness for latency. In AI apps it also cuts LLM token cost."*

---

## 11. Message Broker, Producer/Consumer, Pub/Sub

**Message broker:** Middleware that receives messages from senders and delivers them to receivers. It **decouples** services, absorbs traffic spikes, and enables async communication.

**Roles:**
- **Producer/Publisher:** sends messages.
- **Broker:** stores and routes them.
- **Consumer/Subscriber:** receives and processes them.

**Models:**
| Model | Behavior | Example |
|---|---|---|
| **Point-to-point (queue)** | Each message goes to **one** consumer (work distribution) | Order-processing workers |
| **Publish/Subscribe** | Each message goes to **all** interested subscribers | "OrderPlaced" is consumed by Billing, Inventory and Email |

**Benefits:** loose coupling, scalability, resilience (retries, buffering), and async processing.

**Delivery guarantees:** at-most-once, at-least-once (the common default, so make consumers **idempotent**), and exactly-once (hard, needs transactions or idempotence).

**Example:** E-commerce checkout publishes `OrderPlaced`. Payment, Inventory, Notification and Analytics services each consume it independently.

---

## 12. RabbitMQ (Smart Broker / Dumb Consumer)

**Origin:** Created in **2007** by Rabbit Technologies, written in **Erlang**, and implements **AMQP**. It is now maintained by VMware/Broadcom.

**How it works:**
```
Producer → Exchange → (binding/routing key) → Queue → Consumer
```
- **Exchange types:** direct, fanout, topic, headers.
- The **broker is smart:** it routes, tracks which messages each consumer has received, pushes messages to consumers, manages acks/nacks, retries, priorities, TTLs and dead-letter queues, and **removes messages after acknowledgement**.
- The **consumer is "dumb":** it just receives and acks.

**Best for:** complex routing, task queues, request/reply, low-latency per-message delivery, and background jobs (email, image resize).

**Limits:** No easy replay of old messages, and lower throughput than Kafka at massive scale.

---

## 13. Apache Kafka (Dumb Broker / Smart Consumer)

**Origin:** Built at **LinkedIn** by **Jay Kreps, Neha Narkhede and Jun Rao** (2010), open-sourced in 2011 and later became Apache Kafka. They later founded **Confluent**. It is named after the writer Franz Kafka.

**How it works:** Kafka is a **distributed, partitioned, replicated commit log**.
```
Producer → Topic (Partition 0,1,2...) → Consumer Group
```
- **Topic** is split into **partitions**. Order is guaranteed **within a partition**.
- Messages are **appended to a log and retained** (e.g., 7 days) even after being read.
- The **broker is "dumb":** it mostly appends and serves data, with no per-consumer tracking or complex routing.
- The **consumer is "smart":** it **pulls** data and tracks its own **offset**, so it can **rewind and replay** at will.
- **Consumer groups:** partitions are shared among consumers in a group for parallelism, and each group gets the full stream independently.
- Replication factor (commonly 3) provides fault tolerance. Newer Kafka versions use **KRaft** instead of ZooKeeper for metadata.
- Ecosystem: Kafka Connect, Kafka Streams, Schema Registry.

**Best for:** event streaming, log aggregation, CDC pipelines, real-time analytics, event sourcing and very high throughput (millions of messages per second).

| | RabbitMQ | Kafka |
|---|---|---|
| Broker intelligence | Smart | Dumb (log) |
| Consumer | Dumb, broker **pushes** | Smart, **pulls** and tracks offset |
| Message after consumption | Deleted | Retained, replayable |
| Ordering | Per queue | Per partition |
| Throughput | Moderate | Very high |
| Routing | Rich (exchanges) | Simple (topics/partitions) |
| Use | Task queues, workflows | Event streaming, pipelines |

*Note: "smart/dumb" is a design-philosophy shorthand. Kafka brokers still do real work such as replication and partition leadership. Say it as a mental model.*

> 🎤 *"RabbitMQ is a post office that delivers and forgets. Kafka is a library of events that anyone can re-read."*

---

# PART 6: DATA STORAGE CONCEPTS

## 14. Row-Based vs Column-Based Storage

```
Table: ID | Name | Age | Salary

Row store:    [1,A,30,50k][2,B,40,60k][3,C,25,45k]
Column store: [1,2,3][A,B,C][30,40,25][50k,60k,45k]
```

| | Row-based | Column-based |
|---|---|---|
| Fast for | Reading/writing a **whole record** | Aggregating **few columns** over many rows |
| Workload | OLTP | OLAP |
| Compression | Weaker | Excellent (similar values together, encoding such as RLE and dictionary) |
| Writes | Fast | Slower, usually batched |
| Examples | PostgreSQL, MySQL, Oracle | Redshift, BigQuery, Snowflake, ClickHouse, Parquet, ORC |

**Origin:** The column-store idea was formalized in academic work by Stonebraker et al. (C-Store, 2005, which became Vertica). **Parquet** (Twitter and Cloudera, 2013) and **ORC** (Hortonworks, Facebook, 2013) brought it to big data.

**Example:** `SELECT AVG(salary) FROM employees` over a billion rows. A column store reads only the `salary` column, while a row store reads everything.

---

## 15. Clustering and Liquid Clustering

**Clustering:** Physically co-locating rows with similar values of chosen columns, so queries can **skip** irrelevant files or blocks (data skipping).
- *Clustered index (RDBMS):* the table's rows are physically ordered by key (e.g., InnoDB primary key).
- *Warehouse clustering:* Snowflake clustering keys and BigQuery clustered tables.
- *Delta Lake Z-ORDER:* multi-dimensional clustering using a space-filling curve.

**Liquid Clustering (Databricks / Delta Lake):**
- Introduced by Databricks in **2023** to replace **Hive-style partitioning and Z-ORDER**.
- You declare clustering keys with `CLUSTER BY`. It is **incremental** (only new or unclustered data is reorganized at `OPTIMIZE`).
- You can **change clustering keys** without rewriting all the data.
- It avoids over-partitioning and the small-files problem, and handles skew well.

```sql
CREATE TABLE sales (id INT, region STRING, dt DATE)
CLUSTER BY (region, dt);
OPTIMIZE sales;
```

> 🎤 *"Partitioning needs you to guess the right key up front. Liquid clustering is adaptive and flexible."*

---

## 16. Indexing

**What:** A separate data structure that lets the DB find rows without a **full table scan**, like a book's index. The trade-off is faster reads but slower writes and extra storage.

| Index | Structure | Good for | Bad for |
|---|---|---|---|
| **B-Tree** | Balanced tree, data in all nodes | Equality + range, sorting | Very write-heavy workloads |
| **B+ Tree** | Data only in **leaf nodes**, leaves **linked** | Range scans, disk-friendly (high fan-out) | Default in MySQL InnoDB, PostgreSQL, Oracle |
| **Hash index** | Hash function → bucket | O(1) **equality** lookups | Range queries, sorting |
| **Bitmap** | Bit per value | Low-cardinality columns in DW | High-cardinality, frequent updates |
| **Inverted index** | Term → documents | Full-text search (Elasticsearch/Lucene) | N/A |
| **LSM Tree** | Memtable + sorted files merged | Write-heavy (Cassandra, RocksDB) | Read amplification |

**Origin:** **B-Tree** was invented by **Rudolf Bayer and Edward McCreight at Boeing in 1970**. B+ Tree is a later refinement. LSM trees came from O'Neil et al. (1996).

**Other terms:** clustered vs non-clustered (secondary), composite index (column order matters, "leftmost prefix"), covering index, and `EXPLAIN` to inspect query plans.

**Example:** `CREATE INDEX idx_email ON users(email);` turns an O(n) scan into O(log n).

> 🎤 *"B+ trees dominate because they keep the tree shallow (3–4 levels for millions of rows) and the linked leaves make range queries efficient."*

---

## 17. Partitioning and Sharding

| | Partitioning | Sharding |
|---|---|---|
| Meaning | Splitting a table into pieces | Distributing pieces across **multiple servers** |
| Location | Usually **same** database/server | **Different** nodes |
| Goal | Query pruning, manageability, I/O | **Horizontal scale** of storage and load |
| Complexity | Low | High (routing, rebalancing, cross-shard joins) |

**Types:**
- **Horizontal:** split by rows. **Vertical:** split by columns.
- **Strategies:** **range** (by date), **hash** (even distribution), **list** (by region), **composite**.

**Shard key matters:** a poor key causes **hotspots** and uneven data. **Consistent hashing** (Karger et al., MIT, 1997; used in Dynamo) reduces data movement when nodes are added or removed.

**Examples:**
- *Partitioning:* A Hive/Spark table partitioned by `year/month/day`. A query for one day reads only that folder.
- *Sharding:* Instagram sharded PostgreSQL by user ID. MongoDB and Cassandra shard natively.

**Challenges:** cross-shard joins and transactions, resharding, and hot keys.

> 🎤 *"Partitioning organizes data. Sharding scales it out. Shard-key choice is the most important decision."*

---

# PART 7: DATA MANAGEMENT CONCEPTS

## 18. ACID Transactions

A **transaction** is a group of operations treated as a single unit.

| Letter | Meaning | Example (bank transfer ₹500 A → B) |
|---|---|---|
| **A**tomicity | All or nothing | Debit and credit both happen, or neither |
| **C**onsistency | DB moves from one valid state to another (constraints hold) | Total money unchanged, no negative balance |
| **I**solation | Concurrent transactions don't interfere | Two transfers don't corrupt each other |
| **D**urability | Committed data survives crashes | Survives a power cut (WAL, write-ahead log) |

**Origin:** Jim Gray formalized transactions (late 1970s–1981). **Haerder and Reuter coined "ACID" in 1983.**

**Isolation levels and anomalies:**
| Level | Prevents |
|---|---|
| Read Uncommitted | Nothing (dirty reads possible) |
| Read Committed | Dirty reads |
| Repeatable Read | + Non-repeatable reads |
| Serializable | + Phantom reads (strictest) |

**Contrast, BASE:** *Basically Available, Soft state, Eventually consistent* is the model of many NoSQL systems.

**Where ACID matters:** payments, inventory, bookings. Lakehouse formats (Delta, Iceberg) also bring ACID to data lakes.

---

## 19. Concurrency Control

**Problem:** Simultaneous reads and writes cause lost updates, dirty reads and inconsistent data.

| Mechanism | How it works | Notes |
|---|---|---|
| **Pessimistic locking** (shared/exclusive, **2PL**) | Lock before access | Safe, but risks **deadlocks** and waiting |
| **Timestamp ordering** | Transactions ordered by timestamp, and conflicting operations are rejected or rolled back | No locks |
| **Optimistic concurrency control (OCC)** | Work freely, validate at commit (version column) | Good when conflicts are rare |
| **MVCC** (Multi-Version Concurrency Control) | Keep multiple row versions. Readers see a snapshot, **readers don't block writers** | PostgreSQL, Oracle, MySQL InnoDB, Delta Lake |

**Origin:** Two-phase locking (Eswaran, Gray et al., 1976), MVCC (David Reed, 1978), OCC (Kung and Robinson, 1981).

**Deadlock:** Transaction A waits for B, and B waits for A. The DB detects it and aborts one.

**Example:** Two users book the last seat. Row-level locking, `SELECT ... FOR UPDATE`, or a version check ensures only one succeeds.

---

## 20. Replication

**What:** Maintaining copies of the same data on multiple nodes. The goals are **high availability, fault tolerance, read scaling and disaster recovery**.

| Aspect | Options |
|---|---|
| **Sync** | Primary waits for replica ack. Strong consistency, higher latency, and a stalled replica can block writes |
| **Async** | Primary doesn't wait. Fast, but **replication lag** and possible data loss on failover |
| **Semi-sync** | Waits for at least one replica |
| **Topology** | **Leader-follower** (single writer), **multi-leader**, **leaderless** (quorum reads/writes, as in Dynamo and Cassandra) |

**Quorum rule:** `R + W > N` gives strong consistency in leaderless systems.

**Examples:** MySQL or PostgreSQL primary with read replicas. Kafka topic replication factor 3. S3 cross-region replication. MongoDB replica sets.

**Replication vs Sharding vs Backup:**
- Replication = **copies** (availability, reads)
- Sharding = **splits** (scale writes and storage)
- Backup = **point-in-time recovery** (protects against human error). A replica also copies a bad delete, so it is not a backup.

**Related:** CDC (Change Data Capture, e.g., Debezium) streams DB changes to Kafka, which is how OLTP data reaches the warehouse.

---

## 21. Normalization

**What:** Organizing tables to **reduce redundancy** and avoid update, insert and delete anomalies.

**Origin:** **Edgar Codd**, 1970 (1NF), then 2NF and 3NF in 1971–72. **Boyce-Codd Normal Form (BCNF)** in 1974.

| Form | Rule | Quick example |
|---|---|---|
| **1NF** | Atomic values, no repeating groups | Don't store `phones = "123, 456"` in one cell |
| **2NF** | 1NF + no partial dependency on part of a composite key | Product name shouldn't sit in an `order_items(order_id, product_id)` table |
| **3NF** | 2NF + no transitive dependency | Store `city` → `zip` in a separate table, not with each customer |
| **BCNF** | Every determinant is a candidate key | Stricter 3NF |

**Example:**
- *Unnormalized:* `orders(order_id, customer_name, customer_email, product, price)`. The customer data repeats on every order.
- *Normalized:* `customers`, `orders` and `products` tables linked by foreign keys.

**Trade-off:** More joins mean slower reads. So:
- **OLTP → normalized (3NF)**
- **OLAP → denormalized** (star schema with a fact table and dimension tables, per Kimball) for fast analytics.

---

# QUICK REVISION: HOW IT ALL CONNECTS

```
Sources (apps, IoT, logs)
   │  CDC / Kafka (streaming) or batch extract
   ▼
Data Lake / Lakehouse  ── Bronze → Silver → Gold  (Spark / dbt, ELT)
   │                               │
   ▼                               ▼
Warehouse / BI (OLAP)       ML features, embeddings → Vector DB → RAG / LLM apps
        ▲
   Redis cache in front of serving layers
```

## One-line cheat sheet

| Topic | One-liner |
|---|---|
| Big data | Too big, fast or varied for one machine, so go distributed |
| ETL/ELT | Transform before loading vs after loading in the warehouse |
| Batch/Stream | Scheduled bounded data vs continuous real-time events |
| Warehouse/Lake | Curated structured analytics vs cheap raw storage |
| Medallion | Bronze (raw) → Silver (clean) → Gold (business-ready) |
| RDBMS/NoSQL/Graph/Vector | Tables+ACID / flexible scale / relationships / similarity |
| OLTP/OLAP | Run the business (row, normalized) vs analyze it (column, denormalized) |
| MapReduce/Spark | Disk-based map-shuffle-reduce vs in-memory DAG engine |
| Redis | In-memory store for sub-ms caching, sessions and rate limits |
| RabbitMQ/Kafka | Smart broker pushes and deletes vs dumb log where consumers pull and replay |
| Row/Column | Whole-record access vs column aggregation |
| Indexing | B+ tree for range and equality, hash for equality only |
| Partition/Shard | Split data logically vs distribute across servers |
| ACID | Atomic, Consistent, Isolated, Durable transactions |
| Concurrency | Locks, timestamps, OCC, MVCC |
| Replication | Copies for availability and read scale |
| Normalization | Remove redundancy (OLTP). Denormalize for analytics (OLAP) |
