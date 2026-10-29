const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loadingText");
const welcome = document.getElementById("welcome");
const pageTitle = document.getElementById("pageTitle");
const chatModeButton = document.getElementById("chatMode");
const symptomModeButton = document.getElementById("symptomMode");
const pdfFile = document.getElementById("pdfFile");
const uploadStatus = document.getElementById("uploadStatus");

let currentMode = "chat";
let busy = false;

function setMode(mode) {
    currentMode = mode;

    chatModeButton.classList.toggle("active", mode === "chat");
    symptomModeButton.classList.toggle("active", mode === "symptom");

    pageTitle.textContent =
        mode === "symptom" ? "Symptom Checker" : "Medical Chat";

    input.placeholder = mode === "symptom"
        ? "Describe symptoms, when they started, and how severe they are..."
        : "Ask a health question...";

    loadingText.textContent = mode === "symptom"
        ? "Reviewing your symptom information..."
        : "MedicAI is thinking...";
}

function addMessage(text, role, sources = []) {
    welcome.hidden = true;

    const bubble = document.createElement("article");
    bubble.className = `message ${role}`;

    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = role === "user" ? "YOU" : "MEDICAI";

    const content = document.createElement("div");
    content.textContent = text;

    bubble.append(label, content);

    if (sources.length > 0) {
        const sourceBox = document.createElement("div");
        sourceBox.className = "sources";

        const title = document.createElement("strong");
        title.textContent = "Reference sources";
        sourceBox.appendChild(title);

        sources.forEach((source) => {
            const details = document.createElement("details");
            const summary = document.createElement("summary");

            summary.textContent = [
                source.topic || source.source || "Medical reference",
                source.page !== undefined && source.page !== ""
                    ? ` · Page ${source.page}`
                    : ""
            ].join("");

            const snippet = document.createElement("p");
            snippet.textContent = source.snippet || source.source || "";

            details.append(summary, snippet);
            sourceBox.appendChild(details);
        });

        bubble.appendChild(sourceBox);
    }

    messages.appendChild(bubble);
    bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function postJSON(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status})`);
    }

    return data;
}

async function sendMessage(text) {
    const message = text.trim();

    if (!message || busy) return;

    addMessage(message, "user");
    input.value = "";
    busy = true;
    sendButton.disabled = true;
    loading.hidden = false;

    try {
        const data = await postJSON("/chat", {
            message,
            mode: currentMode
        });

        if (data.error) {
            throw new Error(data.error);
        }

        addMessage(
            data.response || "No response was returned.",
            "bot",
            data.sources || []
        );
    } catch (error) {
        addMessage(
            `Could not get a response: ${error.message}. Check that the Flask server and Ollama are running.`,
            "error"
        );
    } finally {
        busy = false;
        sendButton.disabled = false;
        loading.hidden = true;
        input.focus();
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(input.value);
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

chatModeButton.addEventListener("click", () => setMode("chat"));
symptomModeButton.addEventListener("click", () => setMode("symptom"));

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        setMode("chat");
        sendMessage(button.dataset.prompt || "");
    });
});

document.getElementById("clearChat").addEventListener("click", async () => {
    if (busy) return;

    try {
        const response = await fetch("/clear", { method: "POST" });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Could not clear chat.");

        messages.replaceChildren();
        welcome.hidden = false;
        input.value = "";
    } catch (error) {
        addMessage(error.message, "error");
    }
});

pdfFile.addEventListener("change", async () => {
    const file = pdfFile.files[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        uploadStatus.textContent = "Please select a PDF file.";
        pdfFile.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadStatus.textContent = "Uploading and processing PDF...";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Upload failed.");
        }

        uploadStatus.textContent = data.message || "PDF uploaded.";
    } catch (error) {
        uploadStatus.textContent = `Upload failed: ${error.message}`;
    } finally {
        pdfFile.value = "";
    }
});

document.getElementById("clearKnowledge").addEventListener("click", async () => {
    const confirmed = window.confirm(
        "Reset the knowledge base? Uploaded document entries will be removed."
    );

    if (!confirmed) return;

    uploadStatus.textContent = "Resetting knowledge base...";

    try {
        const response = await fetch("/clear_kb", { method: "POST" });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Reset failed.");
        }

        uploadStatus.textContent = data.message || "Knowledge base reset.";
    } catch (error) {
        uploadStatus.textContent = `Reset failed: ${error.message}`;
    }
});

const groupButton = document.getElementById("groupDetailsButton");
const groupModal = document.getElementById("groupDetailsModal");
const closeGroupButton = document.getElementById("closeGroupDetails");

groupButton?.addEventListener("click", () => {
  groupModal.hidden = false;
});

closeGroupButton?.addEventListener("click", () => {
  groupModal.hidden = true;
});

groupModal?.addEventListener("click", (event) => {
  if (event.target === groupModal) {
    groupModal.hidden = true;
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && groupModal) {
    groupModal.hidden = true;
  }
});