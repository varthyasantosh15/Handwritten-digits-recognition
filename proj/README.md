# DigitLab — Handwritten Digit Recognition with a CNN

A complete, end-to-end digit recognition product: draw a digit on a canvas or upload a
photo of one, and a 2-block convolutional neural network trained on MNIST classifies it
at **99.31% test accuracy**. Every inference is persisted to PostgreSQL with its 28×28
tensor, softmax distribution and latency, so the app doubles as a labelled evaluation log.

No hosted ML service, no TensorFlow.js, no Python at runtime. The network was trained
from scratch in JavaScript and the frozen weights are served by a small TypeScript
inference engine.

## Stack

| Layer | Choice |
| --- | --- |
| Framework | Next.js 16 (App Router, React 19, Server Components) |
| Styling | Tailwind CSS v4 with a custom design system |
| Database | PostgreSQL via Drizzle ORM (`drizzle-orm/node-postgres`) |
| Model | 2-Block CNN (`Conv2D 16×5×5 → MaxPool → Conv2D 32×3×3 → MaxPool → Dense 64 → Softmax 10`, 56,970 params) |
| Training | `scripts/train-cnn.mjs` — pure Node: forward pass, backprop, momentum SGD + cosine LR + shift augmentation |
| Inference | `src/lib/cnn.ts` — typed-array convolution/softmax + 5-crop Test-Time Augmentation |

## Features

- **Draw mode** — 280×280 pressure-free canvas with brush size, undo and clear; auto-read
  fires shortly after you lift the pen.
- **Upload mode** — PNG/JPG/WEBP photos are letterboxed, inverted automatically when the
  paper is light, thresholded with Otsu, cropped to the ink bounding box, downscaled to a
  20 px glyph with 4×4 supersampled area integration, centred on its centre of mass, and
  peak-normalised to 255.
- **Live verdict** — predicted digit, confidence, margin, full softmax histogram, latency.
- **Inside the network** — the preprocessed 28×28 tensor plus all 16 pooled 12×12
  convolution activation maps and the full preprocessing pipeline.
- **Unseen MNIST gallery** — one click classifies a held-out test digit and records the
  ground truth automatically.
- **Feedback loop** — mark a read correct or pick the real digit; rows are updated in
  PostgreSQL and rolled into live accuracy stats.
- **History** — filter by source/digit/mismatch, edit ground truth inline, export CSV,
  delete single rows or clear the log.
- **Model card** — architecture, training curves, per-class recall, 10×10 confusion
  matrix, and a live re-evaluation of 60 held-out digits on every page load.

## Requirements

| Requirement | Version | Notes |
| --- | --- | --- |
| Node.js | 20 or newer | `node -v` |
| npm | 10 or newer | ships with Node |
| PostgreSQL | 14 or newer | local install or Docker (see below) |
| Git | any | for cloning |
| Docker (optional) | any | easiest way to run PostgreSQL |

No Python, GPU or external ML service is needed.

## Getting started

```bash
git clone https://github.com/<your-username>/handwritten-digit-recognition-system.git
cd handwritten-digit-recognition-system

# 1. start PostgreSQL (or use your own instance)
docker compose up -d

# 2. configure environment
cp .env.example .env          # Windows: copy .env.example .env

# 3. install, create tables, run
npm install
npm run db:push
npm run dev
```

Open http://localhost:3000.

Quality checks (also run by GitHub Actions on every push):

```bash
npm run typecheck
npm run lint
npm run build
```

The schema and demo data are also bootstrapped idempotently on first request
(`src/lib/seed.ts`), so an empty database self-heals into a working app.

## Deployment

The app needs a Node server and a database, so GitHub Pages will not work. Deploy to
Vercel (or any Node host) with a hosted Postgres such as Neon or Supabase, and set
`DATABASE_URL` in the project environment variables.

## License

MIT, see [LICENSE](LICENSE).

## Reproducing the model

```bash
# 1. fetch the four MNIST idx files (images + labels, train + test) into /tmp/mnist
#    https://storage.googleapis.com/cvdf-datasets/mnist/
# 2. train and export the artifacts
MNIST_DIR=/tmp/mnist EPOCHS=8 TRAIN_LIMIT=60000 TEST_LIMIT=10000 node scripts/train-cnn.mjs
```

Outputs committed under `src/lib/model/`:

- `weights.json` — conv1, conv2, fc1, fc2 weights & biases (56,970 params)
- `metrics.json` — per-epoch train/test loss + accuracy, per-class recall, confusion matrix
- `samples.json` — 60 correctly classified MNIST test digits for the demo galleries

Reference run: **99.31% test accuracy** (only 69 errors on 10,000 held-out digits, test
loss 0.0196), 8 epochs of momentum SGD with cosine learning rate (lr 0.025 → 0.0015,
batch 32, momentum 0.9, L2 decay 1e-4, ±1 px shift augmentation).

## API

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/api/predict` | Run inference. Body: `{ image, width, height, source }` for a grayscale frame, `{ pixels28 }` for a preprocessed tensor, or `{ sampleId }` for a stored MNIST sample. Persists and returns the row + feature maps. |
| `GET` | `/api/predictions` | List reads. Query: `limit`, `source`, `digit`, `mismatch`. |
| `PATCH` | `/api/predictions/:id` | Set/clear ground truth: `{ actualDigit: 0–9 \| null }`. |
| `DELETE` | `/api/predictions/:id` | Delete one row. |
| `DELETE` | `/api/predictions` | Clear the whole log. |
| `GET` | `/api/stats` | Totals, accuracy, confidence/latency averages, per-digit and per-source counts. |
| `GET` | `/api/samples` | The 60 seeded MNIST test digits (base64 tensors). |
| `GET` | `/api/model` | Model card + training history from the database. |
| `GET` | `/api/health` | Database healthcheck. |
