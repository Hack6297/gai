const STORAGE_KEY = "GabeAI-build-50-chats";

const form = document.querySelector("#chatForm");
const promptInput = document.querySelector("#prompt");
const messages = document.querySelector("#messages");
const sourceList = document.querySelector("#sourceList");
const modelStatus = document.querySelector("#modelStatus");
const useSearch = document.querySelector("#useSearch");
const clearBtn = document.querySelector("#clearBtn");
const exampleBtn = document.querySelector("#exampleBtn");
const newChatBtn = document.querySelector("#newChatBtn");
const deleteChatBtn = document.querySelector("#deleteChatBtn");
const chatList = document.querySelector("#chatList");
const clock = document.querySelector("#clock");

let chats = loadChats();
let activeChatId = chats[0]?.id || createChat("New Chat", false).id;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function loadChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(saved) && saved.length) {
      return saved;
    }
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
  }
  return [];
}

function saveChats() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

function createChat(title = "New Chat", persist = true) {
  const chat = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    title,
    createdAt: Date.now(),
    messages: [
      {
        role: "assistant",
        text: "Welcome to build 50. Ask math, weather, news, or anything else.",
      },
    ],
    sources: [],
  };
  chats.unshift(chat);
  if (persist) {
    activeChatId = chat.id;
    saveChats();
    renderAll();
  }
  return chat;
}

function activeChat() {
  return chats.find((chat) => chat.id === activeChatId) || chats[0];
}

function titleFromMessage(message) {
  const clean = message.replace(/\s+/g, " ").trim();
  return clean.length > 28 ? `${clean.slice(0, 28)}...` : clean || "New Chat";
}

function addChatMessage(role, text, extraClass = "") {
  const chat = activeChat();
  chat.messages.push({ role, text, extraClass });
  if (role === "user" && (chat.title === "New Chat" || chat.messages.length <= 3)) {
    chat.title = titleFromMessage(text);
  }
  saveChats();
  renderAll();
}

function setChatSources(sources) {
  const chat = activeChat();
  chat.sources = sources || [];
  saveChats();
  renderSources(chat.sources);
  renderChatList();
}

function renderAll() {
  renderChatList();
  renderMessages();
  renderSources(activeChat()?.sources || []);
}

function renderChatList() {
  chatList.innerHTML = "";
  for (const chat of chats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chat-tab ${chat.id === activeChatId ? "active" : ""}`;
    button.innerHTML = `
      <span>${escapeHtml(chat.title)}</span>
      <small>${chat.messages.length} messages</small>
    `;
    button.addEventListener("click", () => {
      activeChatId = chat.id;
      saveChats();
      renderAll();
    });
    chatList.append(button);
  }
}

function renderMessages() {
  messages.innerHTML = "";
  const chat = activeChat();
  for (const msg of chat.messages) {
    const article = document.createElement("article");
    article.className = `message ${msg.role} ${msg.extraClass || ""}`.trim();
    const name = msg.role === "user" ? "You" : "GabeAI";
    article.innerHTML = `
      <div class="message-name">${name}</div>
      <p>${escapeHtml(msg.text)}</p>
    `;
    messages.append(article);
  }
  messages.scrollTop = messages.scrollHeight;
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
  addChatMessage("user", message);
  addChatMessage("assistant", "Thinking...", "loading");

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
    const chat = activeChat();
    const loadingIndex = chat.messages.findIndex((msg) => msg.extraClass === "loading");
    if (loadingIndex >= 0) {
      chat.messages.splice(loadingIndex, 1);
    }

    if (!response.ok) {
      chat.messages.push({ role: "assistant", text: payload.error || "The server rejected the request." });
      saveChats();
      renderAll();
      return;
    }

    chat.messages.push({ role: "assistant", text: payload.answer || "No answer came back." });
    chat.sources = payload.sources || [];
    saveChats();
    renderAll();
  } catch (error) {
    const chat = activeChat();
    const loadingIndex = chat.messages.findIndex((msg) => msg.extraClass === "loading");
    if (loadingIndex >= 0) {
      chat.messages.splice(loadingIndex, 1);
    }
    chat.messages.push({
      role: "assistant",
      text: "I could not reach the GabeAI server. Start it with: python app.py",
    });
    saveChats();
    renderAll();
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

newChatBtn.addEventListener("click", () => {
  createChat();
  promptInput.focus();
});

deleteChatBtn.addEventListener("click", () => {
  if (chats.length <= 1) {
    chats = [];
    activeChatId = createChat("New Chat", false).id;
  } else {
    chats = chats.filter((chat) => chat.id !== activeChatId);
    activeChatId = chats[0].id;
  }
  saveChats();
  renderAll();
});

clearBtn.addEventListener("click", () => {
  const chat = activeChat();
  chat.messages = [
    {
      role: "assistant",
      text: "Chat cleared. Ask another question and I will answer directly.",
    },
  ];
  chat.sources = [];
  saveChats();
  renderAll();
});

exampleBtn.addEventListener("click", () => {
  const examples = [
    "What is the square root of 144?",
    "What is the weather in Los Angeles?",
    "What is the latest news about technology?",
  ];
  promptInput.value = examples[Math.floor(Math.random() * examples.length)];
  promptInput.focus();
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const status = await response.json();
    const torchText = status.torchAvailable
      ? "PyTorch is available."
      : "Calculator, no-key weather, news, and search are active.";
    modelStatus.textContent = `${status.name} build 50. ${torchText}`;
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

renderAll();
loadHealth();
tickClock();
setInterval(tickClock, 1000);
