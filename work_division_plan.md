# AI Glaucoma Detector — 2-Member Work Division Plan

To ensure efficient execution, the project is divided into two primary roles: **The AI Engine Developer** and **The UI & Visualization Specialist**.

---

## Member A: AI Engine Developer
**Focus:** Data Acquisition, Model Training, and Optimization.

### Tasks:
1.  **Data Preparation:**
    *   Download the ACRIMA dataset from Kaggle.
    *   Write a small script (or manually) split images into `train/`, `val/`, and `test/` folders.
    *   Upload the organized dataset to Google Drive for Colab access.
2.  **Model Training (Google Colab):**
    *   Set up the `train.py` script on Google Colab.
    *   Implement transfer learning using **EfficientNet-B0**.
    *   Perform training and hyperparameter tuning (Learning Rate, Epochs).
    *   Save the best performing model (`glaucoma_model.pth`).
3.  **Model Delivery:**
    *   Provide the trained model file and the training log to Member B.

---

## Member B: UI & Visualization Specialist
**Focus:** Evaluation, Explainability, and Web Demo.

### Tasks:
1.  **Evaluation & Metrics:**
    *   Develop the `evaluate.py` script to run the trained model on the `test/` set.
    *   Generate the **Confusion Matrix**, **ROC Curve**, and calculate **Accuracy/Sensitivity/Specificity**.
2.  **Explainability (Grad-CAM):**
    *   Implement the `gradcam.py` script to generate heatmaps.
    *   Ensure the heatmaps correctly highlight the optic disc area.
3.  **Web App Development (Streamlit):**
    *   Build the `app.py` frontend.
    *   Integrate the model prediction and Grad-CAM visualization into a single, user-friendly interface.
    *   Ensure the local environment is set up correctly (`requirements.txt`).

---

## Collaborative Tasks (Both Members)
1.  **Project Integration:** Test the final Streamlit app together to ensure it works end-to-end.
2.  **Documentation:**
    *   **Member A:** Write the "Methodology" and "Model Training" sections of the report.
    *   **Member B:** Write the "Results," "Visualization," and "User Guide" sections.
3.  **Presentation:** Prepare the demo and slides for the final submission.

---

## Project Timeline (2-Week Accelerated Schedule)

| Phase | Milestone | Deadline | Primary Driver |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Dataset Split & Colab Ready** | Day 2 | Member A |
| **Phase 2** | **Initial Training & evaluate.py Script** | Day 5 | Both |
| **Phase 3** | **Final Model & Grad-CAM Ready** | Day 9 | Both |
| **Phase 4** | **Streamlit App Integrated** | Day 11 | Member B |
| **Phase 5** | **Testing, Results & Final Report** | Day 14 | Both |
