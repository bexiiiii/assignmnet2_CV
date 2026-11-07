// Shared code for every page: consent, navigation, the photo widget, API calls.
const $ = (id) => document.getElementById(id);
const CLASSES = ["thumbs_up", "peace", "stop"];
const nice = (c) => c.replaceAll("_", " ");
const pct = (x) => (x * 100).toFixed(0) + "%";

// ---------------------------------------------------------------- consent
// Kept in sessionStorage: forgotten when the tab is closed.
const Consent = {
  get: () => sessionStorage.getItem("consent") === "yes",
  set(on) {
    if (on) sessionStorage.setItem("consent", "yes");
    else { sessionStorage.removeItem("consent"); sessionStorage.removeItem("photo"); }
  },
};
function requireConsent() {
  if (!Consent.get()) location.replace("index.html");
}

// ---------------------------------------------------------------- nav
function nav(active) {
  const pages = [["index.html", "Start"], ["recognise.html", "Recognise"], ["stress.html", "Stress test"],
                 ["clip.html", "Clip"], ["results.html", "Results"]];
  const links = pages.map(([href, name]) =>
    `<a href="${href}" class="${href === active ? "on" : ""}">${name}</a>`).join("");
  const out = Consent.get() ? `<a href="#" id="withdraw" class="right">Withdraw consent</a>` : "";
  document.body.insertAdjacentHTML("afterbegin",
    `<nav><div class="in"><span class="brand">Hand gestures</span>${links}${out}</div></nav>`);
  const w = $("withdraw");
  if (w) w.addEventListener("click", (e) => { e.preventDefault(); Consent.set(false); location.href = "index.html"; });
}

// ---------------------------------------------------------------- photo widget
// One photo is shared by Recognise, Stress test and Clip. It lives only in this
// browser tab (sessionStorage) and is deleted when consent is withdrawn.
function photoWidget(onChange) {
  const root = $("photo");
  root.innerHTML = `
    <div class="frame"><video playsinline muted></video><img alt="Your photo">
      <p class="hint">No photo yet</p></div>
    <div class="btns">
      <button data-a="cam">Start camera</button>
      <button data-a="snap" disabled>Take photo</button>
      <label class="file">Upload<input type="file" accept="image/*"></label>
    </div>`;
  const video = root.querySelector("video"), img = root.querySelector("img"), hint = root.querySelector(".hint");
  const cam = root.querySelector("[data-a=cam]"), snap = root.querySelector("[data-a=snap]");
  let stream = null;

  const show = (dataUrl) => {
    img.src = dataUrl; img.classList.add("on"); hint.hidden = true;
    sessionStorage.setItem("photo", dataUrl);
    onChange(true);
  };
  const stop = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; video.classList.remove("on"); cam.textContent = "Start camera"; snap.disabled = true;
  };
  cam.onclick = async () => {
    if (stream) return stop();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 720, height: 720 }, audio: false });
      video.srcObject = stream; await video.play();
      video.classList.add("on"); img.classList.remove("on"); hint.hidden = true;
      cam.textContent = "Stop camera"; snap.disabled = false;
    } catch (e) { alert("Could not open the camera: " + e.message); }
  };
  snap.onclick = () => {
    const s = Math.min(video.videoWidth, video.videoHeight);
    const c = document.createElement("canvas");
    c.width = c.height = Math.min(s, 720);
    const g = c.getContext("2d");
    g.translate(c.width, 0); g.scale(-1, 1);  // same mirrored view the user saw
    g.drawImage(video, (video.videoWidth - s) / 2, (video.videoHeight - s) / 2, s, s, 0, 0, c.width, c.height);
    show(c.toDataURL("image/jpeg", 0.9)); stop();
  };
  root.querySelector("input").onchange = (e) => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader(); r.onload = () => { stop(); show(r.result); }; r.readAsDataURL(f);
  };
  const saved = sessionStorage.getItem("photo");
  if (saved) show(saved); else onChange(false);
}

// ---------------------------------------------------------------- API
async function post(url, fields) {
  const blob = await (await fetch(sessionStorage.getItem("photo"))).blob();
  const fd = new FormData();
  fd.append("photo", blob, "photo.jpg");
  for (const [k, v] of Object.entries(fields || {})) fd.append(k, v);
  const r = await fetch(url, { method: "POST", body: fd, headers: { "X-Consent": Consent.get() ? "yes" : "no" } });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}
async function busy(btn, fn) {
  const t = btn.textContent;
  btn.disabled = true; btn.textContent = "Working…";
  try { await fn(); } catch (e) { alert(e.message); }
  btn.textContent = t; btn.disabled = false;
}
// tiny helper so server values never become HTML
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
