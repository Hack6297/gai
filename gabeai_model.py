import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path

from web_search import SearchClient


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
NOISY_WORDS = {"jobs", "login", "employers", "contributors", "advertise", "cookie"}
SKY_COLOR_RE = re.compile(r"\b(color|colour)\b.*\bsky\b|\bsky\b.*\b(color|colour)\b", re.IGNORECASE)
RELEASE_INTENT_RE = re.compile(r"\b(when|release|released|come out|coming out|publication|publish|published|available|on sale)\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:expected|planned|publication|release|pub(?:lication)? date|available|on sale)?\s*:?\s*"
    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}|"
    r"\d{1,2}/\d{1,2}/\d{4}|"
    r"(?:early|mid|late)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4})",
    re.IGNORECASE,
)
RELEASE_WORDS = {"release", "released", "publication", "published", "expected", "planned", "available", "preorder", "pre-order"}


class GabeTokenizer:
    def __init__(self):
        chars = "\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()[]{}<>/-_+=@#$%^&*`~|\\"
        self.id_to_char = list(dict.fromkeys(chars))
        self.char_to_id = {char: idx for idx, char in enumerate(self.id_to_char)}
        self.unk_id = self.char_to_id[" "]

    @property
    def vocab_size(self):
        return len(self.id_to_char)

    def encode(self, text, max_len=256):
        ids = [self.char_to_id.get(char, self.unk_id) for char in text[-max_len:]]
        if not ids:
            ids = [self.unk_id]
        return ids

    def decode(self, ids):
        return "".join(self.id_to_char[int(idx)] for idx in ids)


if torch is not None:
    class TinyGabeTransformer(nn.Module):
        def __init__(self, vocab_size, context_size=256, embed_dim=96, heads=4, layers=2):
            super().__init__()
            self.context_size = context_size
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(context_size, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.norm = nn.LayerNorm(embed_dim)
            self.lm_head = nn.Linear(embed_dim, vocab_size)

        def forward(self, idx):
            idx = idx[:, -self.context_size:]
            positions = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
            x = self.token_embedding(idx) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.ones(idx.shape[1], idx.shape[1], device=idx.device, dtype=torch.bool),
                diagonal=1,
            )
            x = self.blocks(x, mask=causal_mask)
            x = self.norm(x)
            return self.lm_head(x)
else:
    TinyGabeTransformer = None


class GabeAILanguageModel:
    def __init__(self, app_dir):
        self.app_dir = Path(app_dir)
        self.tokenizer = GabeTokenizer()
        self.available = torch is not None
        self.model = None
        self.device = "cpu"
        if self.available:
            self.model = TinyGabeTransformer(self.tokenizer.vocab_size)
            self.model.eval()
            self._load_weights_if_present()

    def _load_weights_if_present(self):
        weights_path = self.app_dir / "gabeai_weights.pt"
        if not weights_path.exists():
            return
        try:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        except Exception:
            self.model = TinyGabeTransformer(self.tokenizer.vocab_size)
            self.model.eval()

    def status(self):
        return {
            "torchAvailable": self.available,
            "model": "TinyGabeTransformer",
            "weights": str(self.app_dir / "gabeai_weights.pt"),
            "weightsLoaded": bool((self.app_dir / "gabeai_weights.pt").exists() and self.available),
        }

    def sample(self, prompt, max_new_tokens=64, temperature=0.85):
        if not self.available or self.model is None:
            return ""

        try:
            encoded = self.tokenizer.encode(prompt, max_len=180)
            idx = torch.tensor([encoded], dtype=torch.long)
            generated = []
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = self.model(idx)[:, -1, :]
                    logits = logits / max(temperature, 0.05)
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, next_id], dim=1)
                    generated.append(int(next_id.item()))
            return self.tokenizer.decode(generated).strip()
        except Exception:
            return ""


class GabeAIEngine:
    def __init__(self, app_dir):
        self.app_dir = Path(app_dir)
        self.model = GabeAILanguageModel(self.app_dir)
        self.search = SearchClient()

    def status(self):
        status = self.model.status()
        status.update(
            {
                "name": "GabeAI",
                "version": "alpha-0",
                "codeVersion": "alpha-0.4-release-date-answers",
                "engineFile": __file__,
                "search": self.search.status(),
            }
        )
        return status

    def answer(self, question, use_search=True, provider="auto", top_k=5):
        clean_question = " ".join(question.split())
        local_answer = self._local_answer(clean_question)
        if local_answer:
            return {
                "answer": local_answer,
                "sources": [],
                "provider": "local",
                "torch": self.model.status(),
                "neuralTrace": "",
            }

        evidence = []
        provider_used = "none"
        search_error = ""

        if use_search:
            try:
                search_query = self._search_query_for_question(clean_question)
                search_top_k = max(8, top_k) if RELEASE_INTENT_RE.search(clean_question) else top_k
                evidence, provider_used = self.search.search_and_read(
                    search_query,
                    provider=provider,
                    top_k=max(1, min(search_top_k, 8)),
                )
            except Exception as exc:
                search_error = str(exc)

        neural_trace = self.model.sample(
            "GabeAI question: " + clean_question + "\nAnswer:",
            max_new_tokens=44,
        )
        answer = self._compose_answer(clean_question, evidence, neural_trace, search_error)

        return {
            "answer": answer,
            "sources": [item.to_dict() for item in evidence],
            "provider": provider_used,
            "torch": self.model.status(),
            "neuralTrace": neural_trace,
        }

    def _compose_answer(self, question, evidence, neural_trace, search_error):
        lower = question.lower()
        if lower in {"who are you", "who are you?", "what are you", "what are you?"}:
            return (
                "I am GabeAI alpha 0, a local Python assistant. I can answer from simple "
                "built-in knowledge and use no-key web search when I need extra context."
            )

        if evidence:
            if RELEASE_INTENT_RE.search(question):
                answer = self._answer_release_date(question, evidence)
            else:
                answer = self._synthesize_from_sources(question, evidence)
        else:
            answer = (
                "I do not know enough to answer that confidently yet. Try asking it in a "
                "more specific way, and I will search for better context."
            )

        if neural_trace:
            answer += (
                "\n\nModel note: "
                + neural_trace[:180].replace("\n", " ").strip()
            )

        if search_error:
            answer += f"\n\nSearch note: {search_error}"
        return answer

    def _local_answer(self, question):
        lower = question.lower().strip()
        if lower in {"who are you", "who are you?", "what are you", "what are you?"}:
            return (
                "I am GabeAI alpha 0, a local Python assistant. I can answer from simple "
                "built-in knowledge and use no-key web search when I need extra context."
            )
        if SKY_COLOR_RE.search(question):
            return "The sky is usually light blue during the day, and black or very dark at night."
        if "what is your name" in lower or "your name" == lower:
            return "My name is GabeAI."
        return ""

    def _search_query_for_question(self, question):
        lower = question.lower()
        if RELEASE_INTENT_RE.search(question):
            if "warrior" in lower and "graphic novel" in lower:
                return "Warriors Graphic Novel The New Prophecy Part One of Three release date"
            return question + " release date publication date"
        return question

    def _answer_release_date(self, question, evidence):
        candidates = self._extract_date_candidates(question, evidence)
        if not candidates:
            return self._synthesize_from_sources(question, evidence)

        candidates.sort(key=lambda row: row["score"], reverse=True)
        best = candidates[0]
        date = best["date"]
        title = self._clean_title(best["title"])
        answer = f"It looks like {title} is expected to release on {date}."

        alternates = []
        for item in candidates[1:]:
            if item["date"].lower() != date.lower() and item["date"].lower() not in {alt.lower() for alt in alternates}:
                if item["estimated"]:
                    alternates.append(item["date"])
            if len(alternates) == 2:
                break
        if alternates:
            answer += " Some retailer listings show an estimated date of " + ", ".join(alternates) + ", so the exact store date may vary."
        return answer

    def _extract_date_candidates(self, question, evidence):
        candidates = []
        for item in evidence:
            text = " ".join(part for part in [item.title, item.snippet, item.page_text[:2500]] if part)
            for match in DATE_RE.finditer(text):
                date = self._normalize_date(match.group(1))
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                context = text[start:end]
                words = set(TOKEN_RE.findall(context.lower()))
                score = 1.0 + len(words.intersection(RELEASE_WORDS))
                host = ""
                try:
                    from urllib.parse import urlparse
                    host = urlparse(item.url).netloc.lower()
                except Exception:
                    host = ""
                if any(domain in host for domain in ["books.apple.com", "books.google", "goodreads.com", "kobo.com"]):
                    score += 3.0
                if any(domain in host for domain in ["lindentreebooks.com", "harpercollins.com", "warriorcats.com"]):
                    score += 2.5
                if any(domain in host for domain in ["g-mart.com", "mutation.store"]):
                    score += 1.0
                if "reddit.com" in host or "wiki" in host:
                    score -= 0.5
                if "estimated" in context.lower() or "subject to change" in context.lower():
                    score -= 0.2
                candidates.append(
                    {
                        "date": date,
                        "title": item.title,
                        "url": item.url,
                        "score": score,
                        "estimated": "estimated" in context.lower() or "subject to change" in context.lower(),
                    }
                )
        return candidates

    def _normalize_date(self, value):
        cleaned = " ".join(value.replace("  ", " ").strip(" .").split())
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed = datetime.strptime(cleaned, fmt)
                return parsed.strftime("%B ") + str(parsed.day) + parsed.strftime(", %Y")
            except ValueError:
                pass
        return cleaned

    def _clean_title(self, title):
        cleaned = re.sub(r"\s*\|\s*.*$", "", title or "").strip()
        return cleaned or "that book"

    def _synthesize_from_sources(self, question, evidence):
        candidates = self._rank_source_sentences(question, evidence, use_page_text=False)
        if len(candidates) < 2:
            candidates = self._rank_source_sentences(question, evidence, use_page_text=True)

        if not candidates:
            return "The sources are related, but they did not expose enough readable text to summarize confidently."

        chosen = []
        seen = set()
        for _, __, sentence in candidates:
            key = sentence.lower()
            if key not in seen:
                chosen.append(sentence)
                seen.add(key)
            if len(chosen) == 3:
                break

        if len(chosen) == 1:
            return chosen[0]
        return " ".join(chosen)

    def _rank_source_sentences(self, question, evidence, use_page_text):
        terms = set(TOKEN_RE.findall(question.lower()))
        candidates = []
        for item in evidence:
            parts = [item.snippet]
            if use_page_text:
                parts.append(item.page_text)
            text = " ".join(part for part in parts if part)
            for index, sentence in enumerate(SENTENCE_RE.split(text)):
                cleaned = " ".join(sentence.split())
                words = TOKEN_RE.findall(cleaned.lower())
                if 45 <= len(cleaned) <= 260 and len(NOISY_WORDS.intersection(words)) < 2:
                    score = len(terms.intersection(words))
                    if re.search(r"\bis (a|an|the|used|built|designed|known)\b", cleaned, re.IGNORECASE):
                        score += 1
                    if score:
                        candidates.append((score, -index, cleaned))

        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return candidates
