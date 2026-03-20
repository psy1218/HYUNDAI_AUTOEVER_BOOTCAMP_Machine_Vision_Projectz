const destinationMap = {
  "1강의실": 4,
  "2강의실": 3,
  "3강의실": 3,
  "4강의실": 2,
  "5강의실": 1,
  "6강의실": 1,
  "7강의실": 1,
  "사무실": 1,
  "1회의실": 2,
  "2회의실": 3,
  "3회의실": 3,
  "4회의실": 3,
  "5회의실": 4,
  "6회의실": 4,
  "7회의실": 4,
  "대회의실": 4,
  "엘리베이터": 6,
  "화장실": 5
};

const destinationSelect = document.getElementById("destinationSelect");
const showBtn = document.getElementById("showBtn");
const startBtn = document.getElementById("startBtn");
const selectedInfo = document.getElementById("selectedInfo");
const resultDiv = document.getElementById("result");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let intervalId = null;
let selectedDestination = "";

let sending = false; // 프레임 중복 방지 

canvas.width = 640;
canvas.height = 360;


// 드롭다운 채우기
function populateDestinations() {
  Object.keys(destinationMap).forEach((destination) => {
    const option = document.createElement("option");
    option.value = destination;
    option.textContent = destination;
    destinationSelect.appendChild(option);
  });
}

// 보기 버튼 클릭
showBtn.addEventListener("click", () => {
  selectedDestination = destinationSelect.value;

  if (!selectedDestination) {
    selectedInfo.innerHTML = "<p>목적지를 먼저 선택하세요.</p>";
    startBtn.style.display = "none";
    return;
  }

  const zone = destinationMap[selectedDestination];

  selectedInfo.innerHTML = `
    <p><strong>선택한 목적지:</strong> ${selectedDestination}</p>
    <p><strong>도착 구역:</strong> ${zone}구역</p>
  `;

  startBtn.style.display = "inline-block";
  resultDiv.innerHTML = "안내 시작을 누르면 카메라 권한을 요청합니다.";
});

// 안내 시작 버튼 클릭
startBtn.addEventListener("click", async () => {
  if (!selectedDestination) {
    alert("목적지를 먼저 선택하세요.");
    return;
  }

  try {
    resultDiv.innerHTML = "<p>카메라 권한 요청 중...</p>";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      resultDiv.innerHTML = "<p>이 브라우저에서는 카메라를 사용할 수 없습니다.</p>";
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });

    video.srcObject = stream;
    video.style.display = "block";

    resultDiv.innerHTML = "<p>카메라 연결 성공. 안내를 시작합니다.</p>";

    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(sendFrame, 800);

  } catch (err) {
    console.error("카메라 접근 오류:", err);
    resultDiv.innerHTML = `<p>카메라 접근 실패: ${err.name} / ${err.message}</p>`;
  }
});

// 프레임 전송
async function sendFrame() {

  if (sending) return;
  sending = true;

  try{

        if (!video.videoWidth || !video.videoHeight) return;
        if (!selectedDestination) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL("image/jpeg", 0.7);

        try {
            const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageData,
                destination: selectedDestination
            })
            });

            const result = await response.json();
            updateUI(result);

        } catch (error) {
            console.error("서버 전송 오류:", error);
            resultDiv.innerHTML = "<p>서버 전송 오류가 발생했습니다.</p>";
        }
    }   finally{

        sending = false;

    }
}

// 결과 표시
function updateUI(result) {
  resultDiv.innerHTML = `
    <p>현재 구역: ${result.zone}</p>
    <p>현재 방향: ${result.direction}</p>
    <p>신뢰도: ${result.confidence}</p>
    <h2>${result.instruction}</h2>
  `;
}

populateDestinations();