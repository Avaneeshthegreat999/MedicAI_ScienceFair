
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

const cameraInput = document.getElementById("cameraInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("imagePreviewContainer");
const removeImageButton = document.getElementById("removeImageButton");

let currentMode = "chat";
let busy = false;
let previewUrl = null;

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

function addMessage(text, role, sources = [], imageUrl = null) {
    welcome.hidden = true;

    const bubble = document.createElement("article");
    bubble.className = `message ${role}`;

    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = role === "user" ? "YOU" : "MEDICAI";

    bubble.appendChild(label);

    if (imageUrl) {
        const img = document.createElement("img");
        img.src = imageUrl;
        img.alt = "Image sent to MedicAI";
        img.style.maxWidth = "220px";
        img.style.maxHeight = "220px";
        img.style.objectFit = "contain";
        img.style.display = "block";
        img.style.borderRadius = "12px";
        img.style.margin = "8px 0";
        bubble.appendChild(img);
    }

    const content = document.createElement("div");
    content.textContent = text;
    bubble.appendChild(content);

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

function clearSelectedImage() {
    cameraInput.value = "";
    imagePreview.removeAttribute("src");
    previewContainer.hidden = true;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

cameraInput.addEventListener("change", () => {
    const file = cameraInput.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please select an image.");
        clearSelectedImage();
        return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    previewContainer.hidden = false;
});

removeImageButton.addEventListener("click", clearSelectedImage);

async function sendMessage(text) {
    const message = text.trim();
    const file = cameraInput.files?.[0] || null;

    if ((!message && !file) || busy) return;

    const sentImageUrl = file ? URL.createObjectURL(file) : null;

    addMessage(
        message || (file ? "Please describe this image." : ""),
        "user",
        [],
        sentImageUrl
    );

    input.value = "";
    busy = true;
    sendButton.disabled = true;
    loading.hidden = false;

    try {
        let data;

        if (file) {
            const formData = new FormData();
            formData.append("image", file);
            formData.append(
                "message",
                message || "Describe this image and explain what you can observe."
            );

            const response = await fetch("/vision", {
                method: "POST",
                body: formData
            });

            data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Image request failed (${response.status})`);
            }

            clearSelectedImage();
        } else {
            data = await postJSON("/chat", {
                message,
                mode: currentMode
            });
        }

        addMessage(
            data.response || "No response was returned.",
            "bot",
            data.sources || []
        );
    } catch (error) {
        addMessage(
            `Could not get a response: ${error.message}`,
            "error"
        );
    } finally {
        if (sentImageUrl) URL.revokeObjectURL(sentImageUrl);
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

        if (!response.ok) throw new Error(data.error || "Upload failed.");

        uploadStatus.textContent = data.message || "PDF uploaded.";
    } catch (error) {
        uploadStatus.textContent = `Upload failed: ${error.message}`;
    } finally {
        pdfFile.value = "";
    }
});

document.getElementById("clearKnowledge").addEventListener("click", async () => {
    if (!window.confirm("Reset the knowledge base? Uploaded document entries will be removed.")) return;

    uploadStatus.textContent = "Resetting knowledge base...";

    try {
        const response = await fetch("/clear_kb", { method: "POST" });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Reset failed.");

        uploadStatus.textContent = data.message || "Knowledge base reset.";
    } catch (error) {
        uploadStatus.textContent = `Reset failed: ${error.message}`;
    }
});

const groupButton = document.getElementById("groupDetailsButton");
const groupModal = document.getElementById("groupDetailsModal");
const closeGroupButton = document.getElementById("closeGroupDetails");

groupButton?.addEventListener("click", () => {
    if (groupModal) groupModal.hidden = false;
});

closeGroupButton?.addEventListener("click", () => {
    if (groupModal) groupModal.hidden = true;
});

groupModal?.addEventListener("click", (event) => {
    if (event.target === groupModal) groupModal.hidden = true;
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && groupModal) groupModal.hidden = true;
});
