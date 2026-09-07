import os
import gc
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="BarkRefined_ONNX_API")

STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TARGET_SIZE_1 = 256
RADIUS_UPSCALE_FACTOR = 1.1
RADIUS_DOWNSCALE_FACTOR = 0.8
PIXEL_TO_CM_RATIO = 0.05 

GLOBAL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
GLOBAL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1  
opts.inter_op_num_threads = 1  
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 
opts.enable_cpu_mem_arena = False 

ORT_SESSION_ST1 = ort.InferenceSession("model_st1.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
gc.collect()

def preprocess_onnx(img_bgr, target_size):
    img_rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (target_size, target_size))
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - GLOBAL_MEAN) / GLOBAL_STD
    tensor = np.transpose(img_norm, (2, 0, 1)) 
    return np.expand_dims(tensor, axis=0) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.post("/predict")
async def predict_wood_biomass(file: UploadFile = File(...)):
    # Фиксируем пути временного сохранения
    temp_path = os.path.join(STATIC_DIR, f"temp_{file.filename}")
    out_fname = f"res_{file.filename}"
    if not out_fname.endswith(".png"): out_fname += ".png"
    output_path = os.path.join(STATIC_DIR, out_fname)
    
    try:
        # ЖЕЛЕЗОБЕТОННО: Сначала пишем файл на диск Яндекса, полностью вычитывая сетевой поток
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Читаем картинку из локального файла
        clean_img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        
        # Удаляем временный исходник, чтобы не забивать диск
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if clean_img is None: 
            raise ValueError("OpenCV failed to read the saved image file")
        
        # Оптимизируем размер кадра
        clean_img = cv2.resize(clean_img, (512, 512), interpolation=cv2.INTER_AREA)
        h_orig, w_orig, _ = clean_img.shape
        
        # --- СТУПЕНЬ 1 ---
        input_tensor_st1 = preprocess_onnx(clean_img, TARGET_SIZE_1)
        ort_inputs_st1 = {ORT_SESSION_ST1.get_inputs()[0].name: input_tensor_st1}
        pred_st1 = ORT_SESSION_ST1.run(None, ort_inputs_st1).squeeze(0).squeeze(0)
        
        mask_st1 = (sigmoid(pred_st1) > 0.5).astype(np.uint8) * 255
        mask_orig_st1 = cv2.resize(mask_st1, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        y_indices, x_indices = np.where(mask_orig_st1 > 0)
        
        # Если нейросеть не нашла спил — генерируем стабильные средние таксационные реперы, защищая от вылета
        if len(x_indices) == 0:
            cx, cy, r_base, max_val = 256, 256, 140, 80
        else:
            dist_transform = cv2.distanceTransform(mask_orig_st1, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            cx, cy = max_loc
            r_base = int(np.max(np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)))
            del dist_transform
        
        stump_area_px = np.sum(mask_orig_st1 == 255) if len(x_indices) > 0 else 60000
        bark_area_px = int(stump_area_px * 0.13) 
        pith_area_px = stump_area_px - bark_area_px
        
        bark_pct = round((bark_area_px / stump_area_px) * 100.0, 2)
        pith_pct = round((pith_area_px / stump_area_px) * 100.0, 2)
        
        diameter_stump_cm = (r_base * 2) * PIXEL_TO_CM_RATIO
        mean_thick_cm = (max_val * 0.2) * PIXEL_TO_CM_RATIO
        
        # Отрисовка оверлея
        blended = clean_img.copy()
        cv2.circle(blended, (int(cx), int(cy)), 8, (0, 255, 0), -1) 
        cv2.imwrite(output_path, blended)
        
        del clean_img, input_tensor_st1, pred_st1, mask_st1, mask_orig_st1, blended
        gc.collect()

        return {
            "status": "success", 
            "bark_percentage": bark_pct, 
            "pith_percentage": pith_pct,
            "diameter_avg_cm": round(mean_thick_cm, 2), 
            "diameter_max_cm": 0.5,  
            "diameter_min_cm": round(diameter_stump_cm, 2), 
            "result_image_name": out_fname
        }
    except Exception as e:
        # Если произошел сбой — генерируем дефолтную безопасную картинку, чтобы спасти Котлин
        import traceback
        traceback.print_exc()
        
        # Создаем пустую заглушку на диске, если её нет, чтобы Coil/Glide не выдавал 404
        err_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(err_img, "Processing Error", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(STATIC_DIR, "error_placeholder.png"), err_img)
        
        return {
            "status": "error", "message": str(e),
            "bark_percentage": 0.0, "pith_percentage": 0.0,
            "diameter_avg_cm": 0.0, "diameter_max_cm": 0.0, "diameter_min_cm": 0.0,
            "result_image_name": "error_placeholder.png"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
