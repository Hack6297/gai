const form = document.querySelector("#chatForm");
const promptInput = document.querySelector("#prompt");
const messages = document.querySelector("#messages");
const sourceList = document.querySelector("#sourceList");
const modelStatus = document.querySelector("#modelStatus");
const useSearch = document.querySelector("#useSearch");
const clearBtn = document.querySelector("#clearBtn");
const exampleBtn = document.querySelector("#exampleBtn");
const clock = document.querySelector("#clock");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMessage(role, text, extraClass = "") {
  const article = document.createElement("article");
  article.className = `message ${role} ${extraClass}`.trim();
  const name = role === "user" ? "You" : "gai";
  article.innerHTML = `
    <div class="message-name">${name}</div>
    <p>${escapeHtml(text)}</p>
  `;
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function renderSources(sources) {
  sourceList.innerHTML = "";
  if (!sources || sources.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No source links yet.";
    sourceList.append(item);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    const title = escapeHtml(source.title || "Untitled");
    const url = escapeHtml(source.url || "#");
    const score = Number(source.score || 0).toFixed(2);
    item.innerHTML = `
      <a href="${url}" target="_blank" rel="noreferrer">${title}</a>
      <span class="score">match score ${score}</span>
    `;
    sourceList.append(item);
  }
}

async function askGabeAI(message) {
  addMessage("user", message);
  const loading = addMessage("assistant", "Searching and thinking...", "loading");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        useSearch: useSearch.checked,
        provider: "duckduckgo",
        topK: 5,
      }),
    });

    const payload = await response.json();
    loading.remove();

    if (!response.ok) {
      addMessage("assistant", payload.error || "The server rejected the request.");
      return;
    }

    addMessage("assistant", payload.answer || "No answer came back.");
    renderSources(payload.sources || []);
  } catch (error) {
    loading.remove();
    addMessage(
      "assistant",
      "I could not reach the gai server. Start it with: python app.py"
    );
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) {
    promptInput.focus();
    return;
  }
  promptInput.value = "";
  askGabeAI(message);
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

clearBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  renderSources([]);
  addMessage(
    "assistant",
    "Chat cleared. Ask another question and I will answer directly."
  );
});

exampleBtn.addEventListener("click", () => {
  promptInput.value = "What color is the sky?";
  promptInput.focus();
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const status = await response.json();
    const torchText = status.torchAvailable
      ? "PyTorch is available."
      : "PyTorch is not installed; direct answers and no-key search are active.";
    modelStatus.textContent = `${status.name} ${status.version}. ${torchText}`;
  } catch (error) {
    modelStatus.textContent = "Server offline. Run python app.py, then refresh.";
  }
}

function tickClock() {
  const now = new Date();
  clock.textContent = now.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

renderSources([]);
loadHealth();
tickClock();
setInterval(tickClock, 1000);
