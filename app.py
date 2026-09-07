import os
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import gc

app = FastAPI(title="BarkRefined_ONNX_API")

STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.enable_cpu_mem_arena = False

TARGET_SIZE_1 = 256
RADIUS_UPSCALE_FACTOR = 1.1
RADIUS_DOWNSCALE_FACTOR = 0.8
PADDING_SCALE = 3.0
PIXEL_TO_CM_RATIO = 0.05 

GLOBAL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
GLOBAL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ORT_SESSION_ST1 = ort.InferenceSession("model_st1.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
gc.collect()
ORT_SESSION_ST2 = ort.InferenceSession("model_st2.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
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
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        clean_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if clean_img is None: return {"status": "error"}
        h_orig, w_orig, _ = clean_img.shape
        
        input_tensor_st1 = preprocess_onnx(clean_img, TARGET_SIZE_1)
        ort_inputs_st1 = {ORT_SESSION_ST1.get_inputs()[0].name: input_tensor_st1}
        pred_st1 = ORT_SESSION_ST1.run(None, ort_inputs_st1)[0].squeeze(0).squeeze(0)
        
        mask_st1 = (sigmoid(pred_st1) > 0.5).astype(np.uint8) * 255
        mask_orig_st1 = cv2.resize(mask_st1, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        y_indices, x_indices = np.where(mask_orig_st1 > 0)
        if len(x_indices) == 0: return {"status": "error"}
        
        dist_transform = cv2.distanceTransform(mask_orig_st1, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        cx, cy = max_loc
        r_base = int(np.max(np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)))
        r_upscale, r_downscale = int(r_base * RADIUS_UPSCALE_FACTOR), int(max_val * RADIUS_DOWNSCALE_FACTOR)
        
        pad_left = int(np.ceil(max(0, -(cx - r_upscale)) * PADDING_SCALE))
        pad_top = int(np.ceil(max(0, -(cy - r_upscale)) * PADDING_SCALE))
        pad_right = int(np.ceil(max(0, (cx + r_upscale) - (w_orig - 1)) * PADDING_SCALE))
        pad_bottom = int(np.ceil(max(0, (cy + r_upscale) - (h_orig - 1)) * PADDING_SCALE))
        
        bg_pixels = clean_img[mask_orig_st1 == 0]
        mean_bgr = np.mean(bg_pixels, axis=0) if len(bg_pixels) > 0 else [128, 128, 128]
        std_bgr = np.std(bg_pixels, axis=0) if len(bg_pixels) > 0 else [30, 30, 30]
        
        padded_img = np.zeros((h_orig + pad_top + pad_bottom, w_orig + pad_left + pad_right, 3), dtype=np.uint8)
        for ch in range(3):
            noise = np.random.normal(mean_bgr[ch], std_bgr[ch], padded_img.shape[:2])
            padded_img[:, :, ch] = np.clip(noise, 0, 255).astype(np.uint8)
        padded_img[pad_top:pad_top + h_orig, pad_left:pad_left + w_orig] = clean_img
        
        pol_w, pol_h = int(r_upscale), int(2 * np.pi * r_upscale)
        polar_raw = cv2.warpPolar(padded_img, (pol_w, pol_h), (float(cx + pad_left), float(cy + pad_top)), float(r_upscale), cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)
        polar_oriented = cv2.rotate(polar_raw[:, int(r_downscale):pol_w], cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar_final = cv2.resize(polar_oriented, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        input_tensor_st2 = preprocess_onnx(polar_final, 256)
        ort_inputs_st2 = {ORT_SESSION_ST2.get_inputs()[0].name: input_tensor_st2}
        pred_st2 = ORT_SESSION_ST2.run(None, ort_inputs_st2)[0].squeeze(0).squeeze(0)
        
        mask_st2 = (sigmoid(pred_st2) > 0.5).astype(np.uint8) * 255
        pol_mask_resized = cv2.resize(mask_st2, (pol_h, int(r_upscale - r_downscale)), interpolation=cv2.INTER_NEAREST)
        
        pol_mask_native = cv2.rotate(pol_mask_resized, cv2.ROTATE_90_CLOCKWISE)
        h_p_nat, w_p_native = pol_mask_native.shape
        y_c, x_c = np.indices((h_orig, w_orig), dtype=np.float32)
        dx, dy = x_c - float(cx), y_c - float(cy)
        r_matrix = np.sqrt(dx**2 + dy**2)
        theta_matrix = np.arctan2(dy, dx)
        theta_matrix = np.where(theta_matrix < 0, theta_matrix + 2 * np.pi, theta_matrix)
        
        c_t = np.where(np.abs(np.cos(theta_matrix)) < 1e-6, 1e-6, np.cos(theta_matrix))
        s_t = np.where(np.abs(np.sin(theta_matrix)) < 1e-6, 1e-6, np.sin(theta_matrix))
        r_border = np.min(np.where(np.array([(0.0-cx)/c_t, (float(w_orig-1)-cx)/c_t, (0.0-cy)/s_t, (float(h_orig-1)-cy)/s_t]) > 0, np.array([(0.0-cx)/c_t, (float(w_orig-1)-cx)/c_t, (0.0-cy)/s_t, (float(h_orig-1)-cy)/s_t]), np.inf), axis=0)
        r_max_dyn = np.minimum(float(r_upscale), r_border)
        
        map_y = (theta_matrix / (2 * np.pi)) * float(h_p_nat)
        map_x = ((r_matrix - float(r_downscale)) / np.where((r_max_dyn - float(r_downscale)) <= 0, 1e-6, r_max_dyn - float(r_downscale))) * float(w_p_native)
        
        inverse_mask = cv2.remap(pol_mask_native, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        inverse_mask[(r_matrix < float(r_downscale)) | (r_matrix > r_max_dyn)] = 0
        
        ray_thicknesses = []
        max_diag = int(np.ceil(np.sqrt(h_orig**2 + w_orig**2)))
        for angle_deg in range(0, 360, 1):
            x_ray = (cx + np.arange(0, max_diag) * np.cos(np.radians(angle_deg))).astype(np.int32)
            y_ray = (cy + np.arange(0, max_diag) * np.sin(np.radians(angle_deg))).astype(np.int32)
            valid = (x_ray >= 0) & (x_ray < w_orig) & (y_ray >= 0) & (y_ray < h_orig)
            ray_thicknesses.append(np.sum(inverse_mask[y_ray[valid], x_ray[valid]] == 255))
            
        bark_area_px = np.sum(inverse_mask == 255)
        stump_area_px = np.sum(mask_orig_st1 == 255)
        pith_area_px = max(0, stump_area_px - bark_area_px)
        
        bark_pct = round((bark_area_px / stump_area_px) * 100.0, 2) if stump_area_px > 0 else 0.0
        pith_pct = round((pith_area_px / stump_area_px) * 100.0, 2) if stump_area_px > 0 else 0.0
        mean_thick_cm = np.mean(ray_thicknesses) * PIXEL_TO_CM_RATIO
        std_thick_cm = np.std(ray_thicknesses) * PIXEL_TO_CM_RATIO
        diameter_stump_cm = (r_base * 2) * PIXEL_TO_CM_RATIO
        
        col_mask = np.zeros_like(clean_img)
        col_mask[inverse_mask > 0] = (0, 0, 255)
        blended = cv2.addWeighted(clean_img, 1.0, col_mask, 0.5, 0)
        cv2.circle(blended, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        
        out_fname = f"res_{file.filename}.png"
        cv2.imwrite(os.path.join(STATIC_DIR, out_fname), blended)

        del clean_img, img_rgb_st1, tensor_st1, mask_st1, mask_orig_st1, dist_transform, padded_img, polar_raw, polar_oriented, polar_final, tensor_st2, mask_st2, pol_mask_resized, pol_mask_native, y_c, x_c, dx, dy, r_matrix, theta_matrix, map_y, map_x, inverse_mask, ray_thicknesses, col_mask, blended
        gc.collect()
        
        return {
            "status": "success", "bark_percentage": bark_pct, "pith_percentage": pith_pct,
            "diameter_avg_cm": round(mean_thick_cm, 2), "diameter_max_cm": round(std_thick_cm, 2),  
            "diameter_min_cm": round(diameter_stump_cm, 2), "result_image_name": out_fname
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
