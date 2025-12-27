console.log("AI Parking Monitoring Dashboard Loaded");

/* ==========================
   📹 영상 데이터
========================== */
const videos = [
  { src: "videos/KakaoTalk_Video_2025-12-19-20-18-13.mp4", status: "ILLEGAL" },
  { src: "videos/KakaoTalk_Video_2025-12-20-00-11-10.mp4", status: "ILLEGAL" },
  { src: "videos/KakaoTalk_Video_2025-12-19-20-17-58.mp4", status: "ILLEGAL" },
  { src: "videos/KakaoTalk_Video_2025-12-19-20-17-33.mp4", status: "ILLEGAL" },
  { src: "videos/KakaoTalk_Video_2025-12-20-00-11-28.mp4", status: "LEGAL" },
];

const grid = document.getElementById("videoGrid");
const modal = document.getElementById("videoModal");
const modalVideo = document.getElementById("modalVideo");
const modalTitle = document.getElementById("modalTitle");
const backBtn = document.getElementById("backBtn");

/* ==========================
   📹 카드 생성
========================== */
videos.forEach((video, idx) => {
  const card = document.createElement("div");
  card.className = "video-card";

  card.innerHTML = `
    <div class="video-title">
      Camera ${idx + 1}
      <span class="status ${video.status.toLowerCase()}">${video.status}</span>
    </div>
    <video src="${video.src}" autoplay loop muted playsinline></video>
  `;

  card.addEventListener("click", () => {
    modal.classList.remove("hidden");
    modalVideo.src = video.src;
    modalTitle.innerText = `Camera ${idx + 1} · ${video.status}`;
    modalVideo.play();
  });

  grid.appendChild(card);
});

/* ==========================
   ⏱ 타임스탬프
========================== */
function updateTimestamp() {
  document.getElementById("timestamp").innerText =
    `⏱ 실시간 분석 중 · ${new Date().toLocaleString("ko-KR")}`;
}
setInterval(updateTimestamp, 1000);
updateTimestamp();

/* ==========================
   🔙 모달 닫기
========================== */
function closeModal() {
  modalVideo.pause();
  modalVideo.src = "";
  modal.classList.add("hidden");
}

backBtn.addEventListener("click", closeModal);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modal.classList.contains("hidden")) {
    closeModal();
  }
});