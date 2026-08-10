/**
 * Dynamic NLP Model Router Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const presetBtns = document.querySelectorAll(".preset-btn");
  const chatForm = document.getElementById("chat-form");
  const promptInput = document.getElementById("prompt-input");
  const chatThread = document.getElementById("chat-thread");

  const samplePresets = {
    translation: "Translate this to Hindi: My name is Sai Kumar.",
    french: "Translate this into French: Machine learning model routing improves accuracy.",
    sentiment: "This new smartphone has an amazing camera and exceptional battery life, but the software is terrible and constantly crashes.",
    summarization: "Please summarize: Artificial intelligence has transformed modern software engineering by introducing dynamic model routing, automated code generation, and intelligent testing workflows. Companies deploy transformer models to optimize compute cost."
  };

  // Tab Navigation
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });

  // Sample Presets Click
  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      if (samplePresets[presetKey]) {
        promptInput.value = samplePresets[presetKey];
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Handle Form Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const promptText = promptInput.value.trim();
    if (!promptText) return;

    // Render User Message
    appendUserCard(promptText);
    promptInput.value = "";

    // Render Loading Indicator
    const loadingId = appendLoadingCard();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptText })
      });

      let data;
      if (response.ok) {
        data = await response.json();
      } else {
        data = fallbackOfflineRouter(promptText);
      }

      removeElement(loadingId);
      appendBotCard(data);

    } catch (err) {
      removeElement(loadingId);
      const fallbackData = fallbackOfflineRouter(promptText);
      appendBotCard(fallbackData);
    }
  });

  function appendUserCard(text) {
    const card = document.createElement("div");
    card.className = "chat-card user-card";
    card.innerHTML = `<strong>User:</strong> ${escapeHtml(text)}`;
    chatThread.appendChild(card);
    scrollToBottom();
  }

  function appendLoadingCard() {
    const id = "loading-" + Date.now();
    const card = document.createElement("div");
    card.id = id;
    card.className = "chat-card bot-card";
    card.innerHTML = `<div><em>Classifying intent and running transformer model...</em></div>`;
    chatThread.appendChild(card);
    scrollToBottom();
    return id;
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendBotCard(data) {
    const card = document.createElement("div");
    const isFallback = data.model_type === "Fallback";
    card.className = `chat-card bot-card ${isFallback ? 'fallback' : ''}`;

    if (!data.intent_detected) {
      card.innerHTML = `
        <div class="bot-meta">
          <div class="line"><span>Intent Status:</span> Unrecognized Intent</div>
        </div>
        <div class="bot-response-text">${escapeHtml(data.response_text)}</div>
      `;
    } else {
      let metaHtml = `
        <div class="bot-meta">
          <div class="line">Detected Task: <span>${escapeHtml(data.detected_task)}</span></div>
          <div class="line">Selected Model: <span>${escapeHtml(data.selected_model)}</span></div>
          <div class="line">Model Type: <span class="${isFallback ? 'status-fallback' : 'status-primary'}">${escapeHtml(data.model_type)}</span></div>
      `;

      if (isFallback && data.fallback_reason) {
        metaHtml += `<div class="line">Fallback Reason: <span>${escapeHtml(data.fallback_reason)}</span></div>`;
      }

      metaHtml += `<div class="line">Latency: <span>${Math.round(data.latency_ms)} ms</span></div></div>`;

      card.innerHTML = `
        ${metaHtml}
        <div class="bot-response-header">Response:</div>
        <div class="bot-response-text">${escapeHtml(data.response_text)}</div>
      `;
    }

    chatThread.appendChild(card);
    scrollToBottom();
  }

  // Offline Router Handler (in case connection to server.py fails)
  function fallbackOfflineRouter(promptText) {
    return {
      intent_detected: false,
      response_text: "Connection to server.py backend lost. Please ensure 'python server.py' is running."
    };
  }

  function scrollToBottom() {
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
