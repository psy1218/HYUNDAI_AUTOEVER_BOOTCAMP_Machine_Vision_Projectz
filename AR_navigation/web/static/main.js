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

const zoneText = document.getElementById("zone");
const directionText = document.getElementById("direction");
const confidenceText = document.getElementById("confidence");
const instructionText = document.getElementById("instruction");

const arrow = document.getElementById("arrow");

let intervalId = null;
let selectedDestination = "";
let sending = false;

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


// 보기 버튼
showBtn.addEventListener("click", () => {

selectedDestination = destinationSelect.value;

if (!selectedDestination) {

selectedInfo.innerHTML = "목적지를 먼저 선택하세요";

startBtn.style.display = "none";
return;

}

const zone = destinationMap[selectedDestination];

selectedInfo.innerHTML = `
선택한 목적지 : ${selectedDestination} <br>
도착 구역 : ${zone}구역
`;

startBtn.style.display = "inline-block";

resultDiv.innerHTML = "안내 시작을 누르면 카메라가 켜집니다.";

});


// 안내 시작
startBtn.addEventListener("click", async () => {

if (!selectedDestination) {

alert("목적지를 먼저 선택하세요");

return;

}

try {

const stream = await navigator.mediaDevices.getUserMedia({

video:{ facingMode:{ ideal:"environment" }},
audio:false

});

video.srcObject = stream;

if(intervalId) clearInterval(intervalId);

intervalId = setInterval(sendFrame,800);

}
catch(err){

console.error(err);

resultDiv.innerHTML="카메라 접근 실패";

}

});


// 프레임 전송
async function sendFrame(){

if(sending) return;

sending=true;

try{

if(!video.videoWidth) return;

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

ctx.drawImage(video,0,0);

const imageData = canvas.toDataURL("image/jpeg",0.7);

const response = await fetch("/predict",{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

image:imageData,

destination:selectedDestination

})

});

const result = await response.json();

updateUI(result);

}
catch(e){

console.error(e);

}

finally{

sending=false;

}

}


// UI 업데이트
function updateUI(result){

zoneText.innerText = result.zone;
directionText.innerText = result.direction;
confidenceText.innerText = result.confidence.toFixed(2);

instructionText.innerText = result.instruction;

updateArrow(result.instruction);

}


// 화살표
function updateArrow(inst){

if(inst=="직진") arrow.innerText="↑";

else if(inst=="좌회전") arrow.innerText="←";

else if(inst=="우회전") arrow.innerText="→";

else if(inst=="후진") arrow.innerText="↓";

else if(inst=="목적지 주변에 도착") arrow.innerText="✔";

}


populateDestinations();