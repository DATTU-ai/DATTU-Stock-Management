# DATTU Stock Management

## What problem does this solve?

Businesses often receive **sales and purchase bills** as PDFs, scans, or mixed documents. Turning that into **usable stock insight** is slow: data is trapped in unstructured files, reconciliation between sales and purchases is manual, and teams still need **Excel** for reporting and downstream work.

This project helps by:

- **Extracting** line items and key fields from bills using AI-assisted parsing (with validation).
- **Converting** processed data into **structured Excel** outputs you can share or archive.
- **Analyzing inventory** (e.g. surplus/deficit signals) from sales vs purchase documents.
- Providing **user authentication** so access to the app is controlled, with data stored for accounts (MongoDB) while document processing remains oriented around practical workflows.

In short, it reduces manual data entry and spreadsheet wrangling for **stock and bill management**, and speeds up getting from raw documents to **actionable numbers**.

---

## Tech stack

| Layer | Technologies |
|--------|----------------|
| **Frontend** | React 18, Vite 5, Tailwind CSS, PostCSS |
| **Backend** | Python 3, FastAPI, Uvicorn |
| **Data & files** | Pandas, NumPy, OpenPyXL, PDFPlumber |
| **AI** | Groq API (via `groq` SDK) for extraction |
| **Auth & DB** | JWT (`python-jose`), bcrypt, Motor (async MongoDB) |
| **Config** | `python-dotenv` |

Node version used in this repo: see `.node-version` (e.g. **18.20.0**).

---

## Prerequisites

- **Python 3.10+** (3.13 is used in development per the project)
- **Node.js** (aligned with `.node-version` is recommended)
- **MongoDB** running locally or a connection string (for login/signup and user data)
- **Groq API key** in `backend/.env` as `GROQ_API_KEY` for AI extraction (required for full document processing)

---

## How to run the backend

1. Open a terminal and go to the backend folder:

   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended), activate it, then install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create `backend/.env` with at least:

   ```env
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB_NAME=dattu_bill
   JWT_SECRET_KEY=your-secure-secret
   GROQ_API_KEY=your-groq-api-key
   ```

4. Start the API (default port **8000**):

   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## How to run the frontend

1. In another terminal, from the project root:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. The dev server runs on **http://localhost:5173** (configured in `vite.config.js`).

The Vite dev server **proxies** API routes (`/auth`, `/process-document`, `/analyze-bills`, `/health`) to `http://localhost:8000`, so keep the backend running alongside.

**Optional:** set `VITE_API_URL` in `frontend/.env` if you point the UI at a deployed API instead of the local proxy.

**Production build:**

```bash
cd frontend
npm run build
npm run preview
```

---

## Project layout (high level)

- `backend/` — FastAPI app, document parsers, AI extraction, Excel generation, inventory analysis, auth routes
- `frontend/` — React + Vite SPA
