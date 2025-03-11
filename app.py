from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# تحميل نموذج YOLO
model = YOLO("yolo_trained_model.pt")

# قائمة الأصناف المسموح بها
allowed_classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

# إنشاء تطبيق Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    predictions = []

    for file in files:
        np_arr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # تشغيل النموذج على الصورة
        results = model.predict(img, conf=0.2)

        highest_class_name = "No object detected or object not in allowed classes"
        max_conf = -1

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0]) * 100  # تحويل مستوى الثقة إلى نسبة مئوية
                class_name = model.names[int(box.cls[0])]

                # التأكد من أن الكائن المكتشف ضمن الأصناف المسموح بها
                if class_name in allowed_classes and conf > max_conf:
                    max_conf = conf
                    highest_class_name = class_name

        # إذا كانت الثقة أكبر من 60%، يتم إرجاع النتيجة
        if max_conf >= 75:
            predictions.append({"class": highest_class_name, "confidence": max_conf})
        else:
            predictions.append({ "Unknown": "The model is unable to predict the class of this image."})

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
