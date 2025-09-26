const categoryMap = {};
const coco = {
  images: [],
  annotations: [],
  categories: []
};

let imageIdCounter = 1;
let annotationIdCounter = 1;

// 클래스 목록 불러오기
fetch('/classes')
  .then(res => res.json())
  .then(categoryData => {
    const selector = document.getElementById('classSelector');
    selector.innerHTML = '';

    categoryData.forEach(cat => {
      const categoryId = cat.id;
      const className = cat.name;

      // 매핑 및 COCO 카테고리 등록
      categoryMap[className] = categoryId;
      coco.categories.push({ id: categoryId, name: className });

      // 셀렉터에 옵션 추가
      const option = document.createElement('option');
      option.value = className;
      option.textContent = className;
      selector.appendChild(option);
    });
  });

function getCategoryName(categoryId) {
  for (const [name, id] of Object.entries(categoryMap)) {
    if (id === categoryId) return name;
  }
  return "unknown";
}

function initCanvas(canvasId, imagePath, labelPath) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const image = new Image();
  const state = {
    canvas,
    ctx,
    image,
    boxes: [],
    scaleRatio: 1,
    offsetX: 0,
    offsetY: 0,
    baseScale: 1,
    zoomFactor: 1.1,
    isDrawing: false,
    isPanning: false,
    startX: 0,
    startY: 0,
    panStartX: 0,
    panStartY: 0,
    imageId: imageIdCounter++
  };

  image.onload = () => {
    const hRatio = canvas.width / image.width;
    const vRatio = canvas.height / image.height;
    state.baseScale = Math.min(hRatio, vRatio);
    state.scaleRatio = state.baseScale;
    state.offsetX = (canvas.width - image.width * state.scaleRatio) / 2;
    state.offsetY = (canvas.height - image.height * state.scaleRatio) / 2;

    coco.images.push({
      id: state.imageId,
      file_name: imagePath.split('/').pop(),
      width: image.width,
      height: image.height
    });

    drawImageAndBoxes(state);

    if (labelPath) {
      fetch(`/assets/${labelPath}`)
        .then(res => res.json())
        .then(data => {
          const annotations = data.annotations || [];
          const imageMeta = data.images?.[0]; // 이미지 크기 정보

          state.boxes = annotations.map(ann => {
            const [x, y, w, h] = ann.bbox;
            return {
              class: getCategoryName(ann.category_id),
              bbox_canvas: [
                x, y, x + w, y + h
              ]
            };
          });

          // COCO 데이터셋에도 반영
          annotations.forEach(ann => {
            coco.annotations.push(ann);
          });

          if (imageMeta) {
            coco.images.push(imageMeta);
          }

          drawBoxes(state);
        })
        .catch(err => {
          console.error("라벨링 로딩 실패:", err);
        });
    }
  };

  // imagePath는 'TS/...' 와 같은 상대 경로이므로, '/assets/'를 앞에 붙여 완전한 URL로 만듭니다.
  image.src = `/assets/${imagePath}`;

  canvas.addEventListener('mousedown', e => {
    if (e.button === 2) {
      state.isPanning = true;
      state.panStartX = e.offsetX;
      state.panStartY = e.offsetY;
    } else {
      state.startX = e.offsetX;
      state.startY = e.offsetY;
      state.isDrawing = true;
    }
  });

  canvas.addEventListener('mousemove', e => {
    if (state.isPanning) {
      const dx = e.offsetX - state.panStartX;
      const dy = e.offsetY - state.panStartY;
      state.offsetX += dx;
      state.offsetY += dy;
      state.panStartX = e.offsetX;
      state.panStartY = e.offsetY;
      drawImageAndBoxes(state);
    }
  });

  canvas.addEventListener('mouseup', e => {
    if (state.isPanning) {
      state.isPanning = false;
      return;
    }
    if (!state.isDrawing) return;
    state.isDrawing = false;

    const endX = e.offsetX;
    const endY = e.offsetY;
    const className = document.getElementById('classSelector').value;

    const x1 = (state.startX - state.offsetX) / state.scaleRatio;
    const y1 = (state.startY - state.offsetY) / state.scaleRatio;
    const x2 = (endX - state.offsetX) / state.scaleRatio;
    const y2 = (endY - state.offsetY) / state.scaleRatio;

    const cocoBox = [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.abs(x2 - x1),
      Math.abs(y2 - y1)
    ];

    coco.annotations.push({
      id: annotationIdCounter++,
      image_id: state.imageId,
      category_id: categoryMap[className],
      bbox: cocoBox,
      area: cocoBox[2] * cocoBox[3],
      iscrowd: 0
    });

    drawImageAndBoxes(state);
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const mouseX = e.offsetX;
    const mouseY = e.offsetY;

    const prevScale = state.scaleRatio;
    state.scaleRatio *= e.deltaY < 0 ? state.zoomFactor : 1 / state.zoomFactor;

    const dx = mouseX - state.offsetX;
    const dy = mouseY - state.offsetY;
    state.offsetX = mouseX - dx * (state.scaleRatio / prevScale);
    state.offsetY = mouseY - dy * (state.scaleRatio / prevScale);

    drawImageAndBoxes(state);
  });

  canvas.addEventListener('contextmenu', e => e.preventDefault());
}

function drawImageAndBoxes(state) {
  const { ctx, canvas, image, offsetX, offsetY, scaleRatio } = state;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, image.width, image.height,
    offsetX, offsetY, image.width * scaleRatio, image.height * scaleRatio);
  drawBoxes(state);
}

function drawBoxes(state) {
  const { ctx, boxes, scaleRatio, offsetX, offsetY } = state;
  boxes.forEach(box => {
    if (!box.bbox_canvas || !Array.isArray(box.bbox_canvas)) return;
    const [x1, y1, x2, y2] = box.bbox_canvas;
    // 이미지 좌표 → 캔버스 좌표 변환
    const cx1 = x1 * scaleRatio + offsetX;
    const cy1 = y1 * scaleRatio + offsetY;
    const cx2 = x2 * scaleRatio + offsetX;
    const cy2 = y2 * scaleRatio + offsetY;

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
    ctx.fillStyle = 'black';
    ctx.font = '14px sans-serif';
    ctx.fillText(box.class, cx1 + 5, cy1 + 15);
  });
}

function saveAnnotations() {
  // index.html에서 저장한 현재 파일 정보를 가져옴
  const currentFile = window.currentFile;
  if (!currentFile) {
    alert("저장할 파일이 선택되지 않았습니다.");
    return;
  }

  fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // coco 데이터와 함께 파일 정보를 보냄
    body: JSON.stringify({ ...coco, ...currentFile })
  }).then(res => res.json())
    .then(data => alert("COCO 형식으로 저장 완료!"));
}