# GemFinder — Jewelry Image Similarity Search

Upload a photo of any jewelry piece and instantly receive the most visually similar products from your catalogue, powered by ResNet50 embeddings and PostgreSQL with pgvector.

---

## Architecture

```
Client → FastAPI → EmbeddingService (ResNet50) → pgvector similarity search → JSON response
                 ↘ S3Service (import only)
```

- **Model**: ResNet50 pretrained on ImageNet, final FC layer removed → 2048-dim L2-normalised vectors
- **Similarity**: Cosine distance via pgvector `<=>` operator with an HNSW index
- **Storage**: PostgreSQL `products` table with a `vector(2048)` column

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Download from python.org |
| PostgreSQL 15+ | With the [pgvector extension](https://github.com/pgvector/pgvector) installed |
| AWS account | S3 bucket needed for the import script |

---

## 1 — PostgreSQL + pgvector setup

### Install pgvector on Windows

The easiest path on Windows is to use the pre-built binaries from pgvector's GitHub releases, or Docker.

**Option A — Docker (recommended for local dev)**

```powershell
docker run -d `
  --name gemfinder-pg `
  -e POSTGRES_PASSWORD=password `
  -e POSTGRES_DB=gemfinder `
  -p 5432:5432 `
  pgvector/pgvector:pg16
```

**Option B — Local PostgreSQL**

1. Install PostgreSQL from https://www.postgresql.org/download/windows/
2. Download the matching pgvector `.zip` from https://github.com/pgvector/pgvector/releases
3. Copy `vector.dll` to `%PGROOT%\lib` and `vector.control` + `vector--*.sql` to `%PGROOT%\share\extension`
4. Restart the PostgreSQL service

Then create the database:

```sql
CREATE DATABASE gemfinder;
-- The pgvector extension is enabled automatically by the app on first start
```

---

## 2 — Project setup

### Create and activate virtual environment

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> **GPU acceleration (optional)**
> Replace the `torch` / `torchvision` lines in `requirements.txt` with the CUDA wheel:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 3 — Configuration

Copy `.env.example` to `.env` and fill in your values:

```powershell
copy .env.example .env
```

Edit `.env`:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/gemfinder

R2_ACCESS_KEY_ID=AKIA...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=my-gemfinder-bucket
R2_REGION=us-east-1
```

**S3 bucket policy** — the bucket must allow public reads for image URLs to work:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-gemfinder-bucket/*"
  }]
}
```

---

## 4 — Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first start the app will:
1. Enable the `vector` extension in PostgreSQL
2. Create the `products` table
3. Build an HNSW index on the embedding column
4. Load ResNet50 weights (~100 MB, downloaded once from torchvision)

Interactive docs: http://localhost:8000/docs

---

## 5 — Import product images

Prepare a `.zip` archive with your jewelry images (flat or in subdirectories):

```
catalog.zip
├── gold-ring-001.jpg
├── silver-necklace-002.png
└── diamond-earring-003.webp
```

Run the import script:

```powershell
python scripts/import_zip.py catalog.zip
```

The script:
- Skips files that are already in the database (safe to re-run)
- Uploads each image to S3
- Generates a ResNet50 embedding
- Stores `(name, image_url, embedding)` in PostgreSQL

---

## 6 — API Reference

### `POST /upload-image`

Upload a query image and get back the 5 most similar products.

**Request**: `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | JPEG, PNG, WebP, BMP, GIF, or TIFF |

**Response** `200 OK`:

```json
{
  "results": [
    {
      "id": 42,
      "name": "gold-ring-001",
      "image_url": "https://bucket.s3.region.amazonaws.com/products/uuid-gold-ring-001.jpg",
      "similarity": 0.9731
    }
  ],
  "query_embedding_dim": 2048
}
```

---

### `GET /health`

```json
{
  "status": "ok",
  "model": "resnet50",
  "embedding_dim": 2048
}
```

---

## Project structure

```
gemfinder/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, middleware
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── database.py              # Async SQLAlchemy engine, session, init_db
│   ├── models.py                # Product ORM model (pgvector column)
│   ├── schemas.py               # Pydantic request / response schemas
│   ├── services/
│   │   ├── embedding_service.py # ResNet50 singleton feature extractor
│   │   ├── s3_service.py        # boto3 S3 upload helpers
│   │   └── image_service.py     # PIL decode, resize, validation helpers
│   └── routes/
│       ├── upload.py            # POST /upload-image
│       └── search.py            # GET /health
├── scripts/
│   └── import_zip.py            # Bulk import from ZIP archive
├── requirements.txt
├── .env.example
└── README.md
```
