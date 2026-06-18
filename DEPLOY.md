# Deploying Sommel-AI to Streamlit Community Cloud

A free, public URL your colleagues can open without installing anything.
End result: `https://<your-name>-sommelai.streamlit.app`. Total time: ~15 minutes.

## Prerequisites

- A GitHub account (free — sign up at github.com if you don't have one)
- Git installed on your Mac (run `git --version` in Terminal; if missing, install from git-scm.com)

## Step 1 — Create a GitHub repo for the project

1. Go to **github.com** → click **New repository** (green button, top-left).
2. Repo name: `sommelai` (or anything you like)
3. Visibility: **Public** (Streamlit Cloud's free tier requires public repos)
4. Do **not** initialize with README, .gitignore, or license — we have those.
5. Click **Create repository**.
6. GitHub shows a page with setup commands. Keep this tab open.

## Step 2 — Push your project to GitHub

Open Terminal and paste these commands, replacing `<your-username>` with your GitHub username:

```bash
cd "/Users/cheng_family/Library/Application Support/Claude/local-agent-mode-sessions/b9739814-9df3-48c1-87bc-b9c1634e0b6e/f01e282d-0c14-49b0-b5e3-dd3382504f91/local_4a6639fc-38b2-4697-abbd-3a2eb5841a8e/outputs/sommelai"

git init
git add .
git commit -m "Initial commit: Sommel-AI wine recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/sommelai.git
git push -u origin main
```

GitHub will ask for credentials. Use a **Personal Access Token** as the password (Settings → Developer settings → Personal access tokens → Generate, scope = `repo`).

The push will upload ~55 MB — that's the wine dataset CSV. The big artifacts (TF-IDF matrix, KNN model) are *not* pushed; they're regenerated on first app launch via the bootstrap in `app.py`.

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and click **Sign in with GitHub**.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in the form:
   - **Repository**: `<your-username>/sommelai`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: choose something like `sommelai-yourname` (this becomes the public link)
4. Click **Deploy**.
5. Watch the logs. First deploy takes ~3 minutes:
   - ~1 minute to install dependencies from `requirements.txt`
   - ~30 seconds for the bootstrap in `app.py` to build the TF-IDF + KNN artifacts
   - ~30 seconds for Streamlit to start

When it's done you'll get a public URL. Share that with your colleagues.

## Step 4 — When you update the code

```bash
cd "/path/to/sommelai"
git add .
git commit -m "Improved food pairings"
git push
```

Streamlit Cloud auto-redeploys within ~1 minute of your push.

## What your colleagues see

They open the URL → the app loads in ~10-20 seconds (cold start if it's been asleep, instant otherwise) → they get the same UI you have locally. No install, no Python, no Git.

Streamlit Cloud puts free apps to sleep after 7 days of inactivity. Any visitor can wake it just by opening the URL (takes ~30 seconds).

## Common issues

| Symptom | Fix |
|---|---|
| Deploy log says *"ModuleNotFoundError"* | Add the package to `requirements.txt` and push again. |
| Deploy log says *"MemoryError"* during training | Free tier has 1 GB RAM. Reduce `max_features=20000` in `src/train.py` to `10000`. |
| App is slow to load the first time | Normal — the bootstrap is training the model. Subsequent loads are sub-second. |
| 404 on the Streamlit URL | The app was deleted or the repo was made private. Redeploy. |
| `git push` rejected — file too large | The CSV is 51 MB, under GitHub's 100 MB limit. If you hit this, check what else is in `git status`. The `.gitignore` should be keeping artifacts out. |

## What gets shared

You're sharing the **code + the Kaggle Wine Reviews CSV**. The CSV is publicly redistributable under Kaggle's standard terms, so this is fine for a class project. If you want a private repo, Streamlit Cloud's paid tier supports that.
