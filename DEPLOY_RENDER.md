# Deploy the PortfolioAgent to Render

One free web service, deployed from GitHub, gives you one shareable URL. That URL carries the
browser UI, the Agent Card and the A2A endpoint - the link you put on your resume.

---

## What Render is

[Render](https://render.com) is a hosting platform that builds and runs your app straight from
a GitHub repo. You describe the service once in `render.yaml` (a **Blueprint**), and every
`git push` redeploys it.

Why it suits this project:

- **It runs FastAPI as-is.** It runs the same `uvicorn` command you use locally - no code
  changes, no container to write.
- **It deploys from GitHub.** Connect the repo once; pushes deploy themselves.
- **Secrets stay out of the repo.** The OpenAI key is pasted into the dashboard, never
  committed.
- **It gives you a public HTTPS URL.** It looks like `https://portfolioagent-xxxx.onrender.com`,
  and the app reads it from `RENDER_EXTERNAL_URL`, so the Agent Card advertises it with no
  config.

### The free plan, in practice

| Fact | What it means for you |
|---|---|
| **Sleeps after 15 minutes without traffic** | The first visit after a sleep takes about a minute to wake. Open your link before an interview. |
| **Every wake is a fresh process** | Memory, pending intro requests, the audit log and the day's spend counter start empty (a visitor's chat history lives in their browser). Set the optional SMTP variables below and every intro request is emailed to you as it arrives, so none is lost. |
| **No persistent disk** | Nothing written at runtime survives a redeploy. This app keeps no files, so nothing is lost. |
| **One worker** (`--workers 1` in `render.yaml`) | All the state lives in one process, so one worker is required, not just cheaper. |
| **512 MB of RAM, 0.1 CPU** | The app refuses a request body over 2 MiB before reading it, caps every text field and bounds its in-memory stores, so a stranger cannot fill the instance. |

---

## What this repo already has for Render

| File | Role |
|---|---|
| `render.yaml` | The Blueprint. It defines one free Python web service with these settings: <ul><li>build: `pip install -r requirements.txt`</li><li>start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`</li><li>health check: `/health`</li><li>`ADMIN_TOKEN` generated for you</li><li>`OPENAI_API_KEY` asked for once</li></ul> |
| `.python-version` | Pins Python 3.12. Render's default is newer than this stack is tested on. |
| `requirements.txt` | Pinned versions - the stack the tests run on. |
| `.gitignore` | Keeps `.env` (your key) out of the repo. |
| `app/config.py` | On Render (`RENDER=true`), **refuses to boot without `ADMIN_TOKEN`**, so the approval gate is never public by accident. |

---

## Deploy it - step by step

### 1. Check the UI calls its own server
At the top of the page script in `index.html`, the active line must be:

```js
const API = window.location.origin;
```

If the line pointing at `localhost` is the active one instead, the deployed page calls your
laptop rather than the server. Swap the two lines before you push.

### 2. Run the checks locally

```bash
pytest -q
python eval_run.py
```

Both must pass; Render only runs what you push.

### 3. Push this folder to its own GitHub repo
The **contents** of this folder go at the repo root - `render.yaml` must sit at the top level.

```bash
git init
git add .
git status          # confirm .env is NOT listed
git commit -m "PortfolioAgent"
git branch -M main
git remote add origin https://github.com/<you>/portfolioagent.git
git push -u origin main
```

### 4. Create the service from the Blueprint
1. Sign in at [dashboard.render.com](https://dashboard.render.com) with GitHub.
2. Click **New → Blueprint**, then pick your `portfolioagent` repo.
3. Render reads `render.yaml` and shows one web service, `portfolioagent`, on the free plan.
4. When asked for `OPENAI_API_KEY`, paste your key there. This is the only place it goes.
5. Click **Apply**. The first build takes a few minutes; watch it in the **Logs** tab.

### 5. Check it is live
Open your service URL, then:

- `https://<your-service>.onrender.com/health` → `"status": "ok"`, both model tiers.
- `https://<your-service>.onrender.com/.well-known/agent-card.json` → the card, with
  `supportedInterfaces[0].url` set to **your** `onrender.com` URL plus `/a2a`.
- `https://<your-service>.onrender.com/` → the UI. It opens on your build cards (one per week,
  from `data/AGENTS.md`); each one opens a read page at `/portfolio/<week>`. Ask a question, then try **Ask over A2A**.

### 6. Find your admin token
`/actions` (your inbox of intro requests), `/approve`, `/audit` and `/memory` need the admin
token. To get it:

1. Open the service's **Environment** tab and copy `ADMIN_TOKEN`.
2. Open `https://<your-service>.onrender.com/admin` and paste it into the **Admin token**
   field. It is kept in that tab only and sent as a bearer token.
3. From curl or another client, send `Authorization: Bearer <token>`.

When a visitor asks the agent to put them in touch, it calls `request_intro` and the request
pauses at the gate; on `/admin`, **Refresh inbox** to approve or reject it. Without the token,
the inbox, approve and the audit log answer "owner only" - honestly, not with an error.

The card is signed with a key derived from `CARD_SIGNING_SEED`, which the Blueprint generates.
Check it: **Fetch Agent Card** on your page should say **✓ signature verified**, and
`/.well-known/jwks.json` should list one key. If you created the service before this setting
existed, add `CARD_SIGNING_SEED` (any long random string) in **Environment** - or sync the
Blueprint - and redeploy.

### 7. Share the link
Put the URL on your resume and LinkedIn. Every `git push` to `main` redeploys it.

---

## Settings you can change (Environment tab)

| Variable | Default | What it does |
|---|---|---|
| `DAILY_BUDGET_USD` | `1.00` | Model spend per UTC day, across every visitor. Then `/ask` and A2A return 429. |
| `RATE_LIMIT_PER_MINUTE` | `30` | POSTs per client per minute. Then 429 with `Retry-After`. |
| `TRUST_FORWARDED_FOR` | `true` | Keys the rate limit on the client IP behind Render's proxy: `True-Client-IP` / `CF-Connecting-IP` if present, else the first `X-Forwarded-For` hop. |
| `SMTP_SENDER` / `SMTP_PASSWORD` | *(unset)* | Optional. A Gmail address and an App Password ([myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords), 2-Step Verification on) - each intro request is emailed to you. Unset = no email. |
| `NOTIFY_EMAIL` | *(unset)* | Optional. Where intro emails go; defaults to `SMTP_SENDER`. |
| `AGENT_BASE_URL` | *(unset)* | Set it only for a custom domain (no trailing `/` needed); otherwise the card uses `RENDER_EXTERNAL_URL`. Don't paste `.env.example` into **Environment** - a localhost value there would override your Render URL. |

Changing a variable restarts the service.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Deploy fails: `ADMIN_TOKEN must be set when deployed` | The service was created by hand, not from the Blueprint | Add `ADMIN_TOKEN` (any long random string) in **Environment**, or recreate the service from the Blueprint. |
| Build fails on a package | Wrong Python version | Make sure `.python-version` (3.12) is committed at the repo root. |
| The UI loads, but every button fails | `index.html` still points at `localhost` | Do step 1, then push. |
| `/ask` returns 502 | The key is wrong or out of credit (a missing key stops the deploy at boot) | Check `OPENAI_API_KEY` in **Environment** and your OpenAI billing page. |
| `/ask` returns 429 `daily_budget_exhausted` | The day's budget is spent | Wait for the next UTC day, or raise `DAILY_BUDGET_USD`. |
| The first request takes about a minute | The free service was asleep | Expected. Open the link a minute before you need it. |
| Everyone seems to share one rate limit | The proxy's client-IP header is not what the app expects | Log the `True-Client-IP`, `CF-Connecting-IP` and `X-Forwarded-For` headers of one request and check which carries your IP; see `client_key()` in `app/guard.py`. |
| An A2A client gets -32009 | It sent no `A2A-Version` header (the spec reads that as 0.3) | Send `A2A-Version: 1.0`. |
| The card advertises `http://localhost:8000` | Not running on Render, or `AGENT_BASE_URL` is set to localhost | Remove `AGENT_BASE_URL` from **Environment**. |

**Never** commit `.env` or paste your key into `render.yaml`. If a key ever lands in a commit,
rotate it in the OpenAI dashboard - deleting the commit is not enough.
