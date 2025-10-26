const imgInput = document.getElementById("imageInput");
const previewImg = document.getElementById("previewImg");
const ocrBtn = document.getElementById("ocrBtn");
const statusEl = document.getElementById("status");
const output = document.getElementById("output");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");

let selectedFile = null;

imgInput.addEventListener("change", e => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    const reader = new FileReader();
    reader.onload = evt => previewImg.src = evt.target.result;
    reader.readAsDataURL(selectedFile);
  }
});

ocrBtn.addEventListener("click", async () => {
  if (!selectedFile) return alert("Select an image first.");
  statusEl.textContent = "⏳ Extracting text...";
  const formData = new FormData();
  formData.append("image", selectedFile);
  const res = await fetch("/ocr", { method: "POST", body: formData });
  const data = await res.json();
  output.value = data.corrected;
  statusEl.textContent = "✅ Done!";
});

copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(output.value);
  alert("Copied to clipboard!");
});

downloadBtn.addEventListener("click", async () => {
  const res = await fetch("/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: output.value })
  });
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ocr_result.docx";
  a.click();
});
