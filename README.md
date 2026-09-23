# Setup 

### 1. Clone and install

```
git clone <repo-url>
cd rsna_abnormality_detection
python -m venv .venv
.venv\Scripts\activate        # if windows
source .venv/bin/activate     # if Mac/Linux
pip install -r requirements.txt
```

### 2. Kaggle auth (pick ONE method)

#### Option A: OAuth login (easiest)

```
kaggle auth login
```

Log in via the browser popup. Done, credentials are cached automatically.

#### Option B: Access token file

1. Go to kaggle.com/settings → API → "Create New Token"
2. Copy the token string
3. Create `~/.kaggle/access_token` (Windows: `C:\Users\<you>\.kaggle\access_token`) containing **only** the raw token, no quotes, no JSON, no spaces. Make sure the file is NOT saved as a txt file

```
mkdir ~/.kaggle          # if it doesn't exist
echo "YOUR_TOKEN_HERE" > ~/.kaggle/access_token
```

**Don't use both methods, and don't create `kaggle.json` alongside `access_token`.** Having two auth files at once causes the CLI to silently fall back to the wrong one. 

```
kaggle config view

```
Should show your chosen option for auth method. If it says LEGACY_API_KEY then make sure to delete kaggle.json. If both methods aren't working (use kaggle competitions list as sanity check) just manually set the API ket for the session using

```
$env:KAGGLE_API_TOKEN=your token
```

### 3. Join the competition

You need your own Kaggle account joined to the competition (one join per person, not shareable). Accept the rules at the competition page before running anything.

### 4. Verify

```
kaggle competitions list
```

If that returns a list without an auth error, you're set. 

# About Submission

This is a code competition, so our final submission is going to be a kaggle notebook and not a csv file. 
We will have a inference notebook that lives on kaggle and uses the data from /kaggle/input/rsna-knee-abnormality-detection/ and the weights from our trained model that lives on kaggle too as a private dataset. We need to make sure we meet the runtime contraints on kaggle and the no internet connection requirement. 
This inference notebook will create a csv file matching sample_submission.csv file exactly.