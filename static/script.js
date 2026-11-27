const form = document.getElementById("summarize-form");
const inputEl = document.getElementById("dialogue-input");
const outputEl = document.getElementById("summary-output");
const submitBtn = document.getElementById("submit-btn");
const copyBtn = document.getElementById("copy-btn");
const sampleBtn = document.getElementById("sample-btn");

const maxLenSlider = document.getElementById("max-len");
const beamSlider = document.getElementById("beam-size");
const lenVal = document.getElementById("len-val");
const beamVal = document.getElementById("beam-val");
const wordCountEl = document.getElementById("word-count");
const summaryWordsEl = document.getElementById("summary-words");
const latencyEl = document.getElementById("latency-info");

const SAMPLE_DIALOGUE = `Amit: Hey, did you check the server logs for the payment gateway?
Priya: Yes, there's a timeout spike around 2 PM right after the new patch went live.
Amit: Let's roll back to v2.4 immediately and monitor the error rate for 15 minutes.
Priya: Rolling it back now. I'll ping you once the dashboard turns green.`;

const countWords = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);

inputEl.addEventListener("input", () => {
    wordCountEl.textContent = `${countWords(inputEl.value)} words`;
});

maxLenSlider.addEventListener("input", () => {
    lenVal.textContent = maxLenSlider.value;
});

beamSlider.addEventListener("input", () => {
    beamVal.textContent = beamSlider.value;
});

sampleBtn.addEventListener("click", () => {
    inputEl.value = SAMPLE_DIALOGUE;
    wordCountEl.textContent = `${countWords(inputEl.value)} words`;
});

copyBtn.addEventListener("click", async () => {
    const text = outputEl.textContent.trim();
    if (!text || outputEl.classList.contains("placeholder")) return;

    await navigator.clipboard.writeText(text);
    copyBtn.textContent = "Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("copied");
    }, 1500);
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const dialogue = inputEl.value.trim();

    if (dialogue.length < 10) {
        latencyEl.textContent = "Please enter at least 10 characters.";
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Summarizing...";
    outputEl.classList.add("placeholder");
    outputEl.textContent = "Running beam search on your custom Transformer...";
    latencyEl.textContent = "Processing...";

    try {
        const res = await fetch("/summarize/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dialogue,
                max_len: Number(maxLenSlider.value),
                beam_size: Number(beamSlider.value),
            }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);

        outputEl.classList.remove("placeholder");
        outputEl.textContent = data.summary || "No summary generated.";
        summaryWordsEl.textContent = `${countWords(data.summary || "")} words`;
        latencyEl.textContent = `Done in ${data.latency_ms} ms (${data.device})`;
    } catch (err) {
        outputEl.classList.remove("placeholder");
        outputEl.textContent = `Error: ${err.message}`;
        latencyEl.textContent = "Request failed";
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Summarize";
    }
});