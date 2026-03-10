# Streamlit_diamente

Diamond price prediction app powered by a trained RandomForest model.

## Quickstart
1. Install deps: `pip install -r requirements.txt`
2. Run locally: `streamlit run streamlit_app.py`
3. Open the provided local URL to use the form and get price estimates.

Model file: `best_randomforest_model.pkl` (stored in the repo root). Make sure it stays alongside `streamlit_app.py`.

## Hugging Face model download from your account
If the model file is not found locally, the app downloads it with `huggingface_hub`.

Set these values so download uses your account/repo:
- `HF_REPO_ID` (example: `your-username/your-model-repo`)
- `HF_TOKEN` (a Hugging Face access token with repo read access)

You can set them either in Streamlit secrets (`.streamlit/secrets.toml`) or env vars.

Example `secrets.toml`:
```toml
HF_REPO_ID = "your-username/your-model-repo"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

Example PowerShell session:
```powershell
$env:HF_REPO_ID="your-username/your-model-repo"
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
streamlit run streamlit_app.py
```
