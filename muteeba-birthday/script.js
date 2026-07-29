/* ═══════════ Muteeba's Birthday — interactions ═══════════ */

// ---------- Confetti ----------
const canvas = document.getElementById("confetti");
const ctx = canvas.getContext("2d");
let pieces = [];
let confettiRunning = false;

function sizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
sizeCanvas();
window.addEventListener("resize", sizeCanvas);

const COLORS = ["#f4b6c2", "#e56b8c", "#f2cf8d", "#d9a05b", "#fff5ec", "#c2497a", "#ffd9e3"];

function spawnConfetti(count, burst) {
  for (let i = 0; i < count; i++) {
    pieces.push({
      x: burst ? canvas.width / 2 + (Math.random() - 0.5) * 200 : Math.random() * canvas.width,
      y: burst ? canvas.height * 0.55 : -20 - Math.random() * canvas.height * 0.5,
      w: 6 + Math.random() * 7,
      h: 8 + Math.random() * 8,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      vy: 1.6 + Math.random() * 2.6,
      vx: burst ? (Math.random() - 0.5) * 9 : (Math.random() - 0.5) * 1.6,
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.22,
      shape: Math.random() < 0.25 ? "circle" : "rect",
    });
  }
  if (!confettiRunning) {
    confettiRunning = true;
    requestAnimationFrame(tick);
  }
}

function tick() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  pieces = pieces.filter((p) => p.y < canvas.height + 30);
  for (const p of pieces) {
    p.x += p.vx;
    p.y += p.vy;
    p.vx *= 0.995;
    p.rot += p.vr;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.fillStyle = p.color;
    if (p.shape === "circle") {
      ctx.beginPath();
      ctx.arc(0, 0, p.w / 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
    }
    ctx.restore();
  }
  if (pieces.length > 0) {
    requestAnimationFrame(tick);
  } else {
    confettiRunning = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// ---------- Intro ----------
const intro = document.getElementById("intro");
const btnOpen = document.getElementById("btnOpen");
document.body.classList.add("locked");

function openSurprise() {
  intro.classList.add("hidden");
  document.body.classList.remove("locked");
  spawnConfetti(220, false);
  setTimeout(() => spawnConfetti(90, false), 900);
}
btnOpen.addEventListener("click", openSurprise);
if (location.hash === "#open") openSurprise();

// ---------- Reveal on scroll ----------
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);
document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));

// ---------- Cake ----------
const cakeEl = document.getElementById("cakeEl");
const cakeSection = document.getElementById("cake");

function blowCandles() {
  if (cakeEl.classList.contains("blown")) return;
  cakeEl.classList.add("blown");
  cakeSection.classList.add("done");
  spawnConfetti(160, true);
}
cakeEl.addEventListener("click", blowCandles);
cakeEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    blowCandles();
  }
});

// ---------- Lightbox ----------
const lightbox = document.createElement("div");
lightbox.className = "lightbox";
const lightboxImg = document.createElement("img");
lightbox.appendChild(lightboxImg);
document.body.appendChild(lightbox);

document.querySelectorAll(".polaroid img").forEach((img) => {
  img.addEventListener("click", () => {
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt;
    lightbox.classList.add("open");
  });
});
lightbox.addEventListener("click", () => lightbox.classList.remove("open"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") lightbox.classList.remove("open");
});
