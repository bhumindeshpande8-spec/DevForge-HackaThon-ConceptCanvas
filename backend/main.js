const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const statusEl = document.getElementById('status');
const codeOutput = document.getElementById('code-output');
const previewVideo = document.getElementById('preview-video');
const presetButtons = document.querySelectorAll('.preset-btn');

// Fill textarea when a preset is clicked
presetButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const text = btn.getAttribute('data-prompt');
    promptInput.value = text || '';
  });
});

// Call backend to generate Manim code + video
generateBtn.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    statusEl.textContent = 'Please enter a description.';
    return;
  }

  statusEl.textContent = 'Generating... this may take some time.';
  codeOutput.textContent = '';
  previewVideo.removeAttribute('src');

  try {
    const res = await fetch('http://localhost:8000/generate_scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    if (!res.ok) {
      statusEl.textContent = 'Backend error: ' + res.status;
      return;
    }

    const data = await res.json();
    codeOutput.textContent = data.manim_code || '# No code returned';

    if (data.video_url) {
      const url = 'http://localhost:8000' + data.video_url;
      previewVideo.src = url;
      previewVideo.load();
    } else {
      statusEl.textContent = 'No video generated (Manim may have failed).';
      return;
    }

    statusEl.textContent = 'Done.';
  } catch (err) {
    statusEl.textContent = 'Request failed: ' + err.message;
  }
});
