const input = document.getElementById("imageInput");
const ocrBtn = document.getElementById("ocrBtn");
const status = document.getElementById("status");
const preview = document.getElementById("preview");
const finalText = document.getElementById("finalText");
const spelledText = document.getElementById("spelledText");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");

let file = null;
input.addEventListener("change", e => {
  file = e.target.files[0];
  if (file) {
    const r = new FileReader();
    r.onload = ev => preview.src = ev.target.result;
    r.readAsDataURL(file);
  }
});

ocrBtn.addEventListener("click", async () => {
  if (!file) return alert("Select an image first");
  status.textContent = "Processing... (may take several seconds on CPU)";
  const fd = new FormData();
  fd.append("image", file);
  try {
    const res = await fetch("/ocr", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) {
      status.textContent = "Error: " + data.error;
      return;
    }
    finalText.value = data.final || "";
    spelledText.value = data.spelled || "";
    status.textContent = "Done";
  } catch (e) {
    status.textContent = "Request failed: " + e.message;
  }
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(finalText.value || "");
  alert("Copied");
});

downloadBtn.addEventListener("click", async () => {
  const res = await fetch("/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: finalText.value || "" })
  });
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ocr_output.docx";
  a.click();
});
