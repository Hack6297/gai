# GabeAI alpha 0

GabeAI is a local Python web app with:

- a Windows XP style chat UI in `index.html`
- `background.jpg` used as the desktop wallpaper
- a tiny PyTorch Transformer scaffold named `TinyGabeTransformer`
- a no-key web search reader that ranks and cites the closest pages
- no Flask or other server framework; `app.py` uses Python's standard library HTTP server

## Run

```powershell
cd "C:\Users\Admin\Documents\GabeAI\GabeAI build alpha 0"
python app.py
```

Open:

```text
http://127.0.0.1:8765
```

## Deploy On Render

GitHub account: `Hack6297`

GitHub repository name:

```text
GabeAI
```

Render build command:

```text
pip install -r requirements.txt
```

Render start command:

```text
python app.py
```

The app reads Render's `PORT` environment variable automatically and listens on `0.0.0.0`, so it is ready for Render Web Services.

## PyTorch

This app imports PyTorch when it is installed. If `torch` is missing, GabeAI still runs the UI and web-search answer path, but neural sampling is disabled.

Install PyTorch in a compatible Python environment:

```powershell
pip install torch
```

The bundled Python here may be newer than current PyTorch wheels. If installation fails, use a Python version supported by PyTorch.

## Search

GabeAI does not require API keys. It uses DuckDuckGo HTML search as a fallback when a question needs current or outside information. Simple questions are answered directly without showing search-result clutter in the chat.

## Files

- `app.py` - local server and API routes
- `gabeai_model.py` - GabeAI engine and tiny PyTorch model
- `web_search.py` - DuckDuckGo search, page text extraction, ranking
- `index.html` - Windows XP style chat screen
- `style.css` - XP desktop layout using `background.jpg`
- `app.js` - browser chat behavior

## Implementation References

- XP.css docs: https://botoxparty.github.io/XP.css/
- XP.css GitHub: https://github.com/botoxparty/XP.css
- PyTorch `TransformerEncoderLayer`: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- DuckDuckGo HTML endpoint reference from SearXNG docs: https://docs.searxng.org/dev/engines/online/duckduckgo.html
