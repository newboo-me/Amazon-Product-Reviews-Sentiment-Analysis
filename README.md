# Amazon-Product-Reviews-Sentiment-Analysis
# Amazon Product Reviews Sentiment Analysis

การสร้างและเปรียบเทียบโมเดล Machine Learning เพื่อจำแนกอารมณ์ของข้อความรีวิวสินค้า (เชิงบวก vs เชิงลบ) จากชุดข้อมูล Amazon Product Reviews โดยเน้นการจัดการข้อมูลไม่สมดุล (Imbalanced Data) และการเปรียบเทียบประสิทธิภาพระหว่างโมเดลกลุ่ม Linear และ Tree-Based

---

**ข้อมูลและการเตรียมข้อมูล (Dataset & Preprocessing)**

* **ชุดข้อมูล:** Amazon Product Reviews จาก Kaggle (สุ่มตัวอย่าง 50,000 แถวเพื่อการประมวลผล)
* **การจัดกลุ่มเป้าหมาย (Sentiment Labeling):**
  * คะแนน 4-5 ดาว กำหนดเป็น **Positive (1)**: 84.29%
  * คะแนน 1-2 ดาว กำหนดเป็น **Negative (0)**: 15.71%
  * คะแนน 3 ดาว ตัดออกจากการทดลองเพื่อลดความคลุมเครือของภาษา
* **การทำความสะอาดข้อความ (Text Cleaning):**
  * ลบ HTML tags ด้วย Regular Expression
  * แปลงข้อความเป็นตัวพิมพ์เล็กทั้งหมด
  * กรองเก็บเฉพาะตัวอักษรภาษาอังกฤษ (a-z) ตัดตัวเลขและอักขระพิเศษ
  * ตัดคำฟุ่มเฟือย (Stopwords) ออก โดยยกเว้นคำปฏิเสธ (`not`, `no`, `nor`, `neither`, `never`) เพื่อรักษาความหมายเชิงลบ

---

**การสกัดคุณลักษณะของข้อความ (Feature Extraction & Selection)**

* ใช้ **TF-IDF Vectorizer** ในการแปลงข้อความเป็นเวกเตอร์ตัวเลข
* กำหนด `ngram_range=(1, 2)` เพื่อจับคู่คำศัพท์แบบ 1 คำ และ 2 คำต่อเนื่อง (เช่น `not good`, `not worth`)
* กำหนด `max_features=5000` เพื่อคัดเลือก 5,000 คำที่มีค่าน้ำหนักสูงสุด
* กำหนด `min_df=5` ตัดคำที่ปรากฏน้อยกว่า 5 ครั้งเพื่อลดสัญญาณรบกวน (Noise)
* แบ่งข้อมูล Train / Test ที่สัดส่วน 80:20 (ควบคุมสัดส่วนคลาสด้วย Stratified Sampling)

---

**ผลการเปรียบเทียบโมเดล (Model Benchmarking)**

ทดสอบ 4 โมเดลบนชุดข้อมูลทดสอบ (Test Set) จำนวน 10,000 แถว โดยวัดผลที่คลาส Negative (0) เป็นหลัก:

| Model | Accuracy | Macro F1 | Neg Precision | Neg Recall | Neg F1-Score | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC (Balanced)** | **92.52%** | **0.8702** | **0.7142** | **0.8733** | **0.7858** | **1.00** |
| **Logistic Regression (Balanced)** | 91.84% | 0.8622 | 0.6848 | **0.8905** | 0.7742 | 0.45 |
| **XGBoost** | 90.02% | 0.8347 | 0.6343 | 0.8612 | 0.7306 | 59.62 |
| **Complement Naive Bayes** | 88.51% | 0.8185 | 0.5890 | 0.8892 | 0.7086 | **0.02** |

**ข้อสรุปจากการทดลอง:**
1. **LinearSVC (Balanced)** ให้ประสิทธิภาพโดยรวมดีที่สุด ทั้งความแม่นยำ (Accuracy 92.52%) และความสมดุลในการดักจับรีวิวเชิงลบ (Neg F1-Score 0.7858)
2. **Logistic Regression (Balanced)** ทำค่า Neg Recall ได้สูงสุดที่ 89.05% สามารถดักจับรีวิวเชิงลบได้มากที่สุด แต่มี Precision ต่ำกว่า LinearSVC เล็กน้อย
3. **Linear Models vs Tree-Based:** LinearSVC และ Logistic Regression ทำงานได้เร็วกว่า XGBoost อย่างชัดเจน (ใช้เวลา 0.45 - 1.00 วินาที เทียบกับ 59.62 วินาที) เนื่องจากโครงสร้างข้อมูลแบบ Sparse Matrix จาก TF-IDF เหมาะกับการแบ่งระนาบแบบเชิงเส้นมากกว่าการแตกกิ่งแบบต้นไม้

---

**คำศัพท์บ่งชี้อารมณ์สำคัญ (Feature Interpretability)**

สกัดจากค่าน้ำหนักสัมประสิทธิ์ (Coefficients) ของโมเดล Logistic Regression:



https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews

<img width="1008" height="294" alt="Screenshot 2026-09-09 201046" src="https://github.com/user-attachments/assets/7d6c0104-b139-4232-b818-0c867bc36333" />
<img width="833" height="157" alt="Screenshot 2026-09-09 201103" src="https://github.com/user-attachments/assets/6d034ed2-9795-4f85-9b84-12d46c95dce7" />
<img width="658" height="245" alt="Screenshot 2026-09-09 201116" src="https://github.com/user-attachments/assets/788c6d12-4432-49ab-9aa7-c25289a73e64" />



* **Top 10 Positive Words:** great, best, good, delicious, excellent, love, loves, perfect, yummy, nice
* **Top 10 Negative Words:** not, not good, disappointed, worst, disappointing, awful, terrible, horrible, not worth, unfortunately
