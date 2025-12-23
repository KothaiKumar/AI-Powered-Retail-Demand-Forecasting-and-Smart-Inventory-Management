## AI-Powered Retail Product Classification and Time-Series Demand Forecasting for Smart Inventory Management
The project focuses on developing an intelligent retail inventory management system that automates product identification and predicts future demand using Artificial Intelligence. By integrating deep learning–based image classification with time-series forecasting, the system helps retailers make proactive inventory decisions and reduce stock imbalances.

## About
AI-Powered Retail Product Classification and Demand Forecasting is a smart inventory management solution designed to address challenges such as manual product identification, inaccurate demand estimation, and inefficient stock replenishment. Traditional inventory systems rely heavily on human intervention and basic statistical methods, leading to overstocking or stockouts.

This project integrates a deep learning–based computer vision model (DenseNet121) to classify retail products from images and a statistical time-series forecasting model (Prophet) to predict future demand trends. The system automatically identifies product categories, analyzes historical demand patterns, forecasts upcoming demand, and computes reorder recommendations using safety stock and lead-time analysis. The solution improves inventory accuracy, reduces operational effort, and supports data-driven decision-making in modern retail environments.

## Features
1. Automated retail product classification using deep learning.

2. Time-series demand forecasting for proactive inventory planning.

3. Integration of computer vision and forecasting models.

4. Automatic reorder point and safety stock calculation.

5. Scalable framework suitable for real-world retail deployment.

6. Reduced manual intervention and improved operational efficiency.

7. Forecast outputs stored as CSV files for analysis and visualization.

8. High reliability even with limited or noisy historical data.

## Requirements
### Operating System
Requires a 64-bit OS (Windows 10 / Windows 11 or Ubuntu) for compatibility with AI frameworks.

### Development Environment
Python 3.8 or later for model execution and system integration.

### Deep Learning Frameworks
TensorFlow and Keras for loading and executing the DenseNet121 model.
Prophet for time-series demand forecasting.

### Image Processing Libraries
OpenCV for image preprocessing, resizing, normalization, and color conversion.

### Data Handling & Analysis
NumPy and Pandas for numerical computation and time-series data processing.

### Visualization Tools
Matplotlib for demand forecast visualization.

### IDE
VS Code or Jupyter Notebook for development, testing, and debugging.

### Additional Dependencies
scikit-learn, cmdstanpy, TensorFlow-GPU (optional for acceleration).

## System Architecture
The system follows a modular architecture consisting of:
1. Image Input Module – Accepts retail product images.
2. Preprocessing Module – Performs resizing, normalization, and augmentation.
3. Product Classification Module – Uses DenseNet121 to identify product category.
4. Demand Forecasting Module – Uses Prophet to predict future demand.
5. Inventory Optimization Module – Calculates reorder point and safety stock.
6. Output Module – Generates CSV forecasts and reorder alerts.

<img width="1000" height="1000" alt="Gemini_Generated_Image_gbv1h5gbv1h5gbv1" src="https://github.com/user-attachments/assets/b80b6af6-1ecf-4527-9a0c-132d6e258ef0" />

## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Product Classification

<img width="638" height="549" alt="Screenshot 2025-12-15 203402" src="https://github.com/user-attachments/assets/7935fc1b-7498-478c-b392-c515be749f8c" />

#### Output2 - Inventory Decision Output
<img width="649" height="811" alt="Screenshot 2025-12-15 203751" src="https://github.com/user-attachments/assets/cd2163bd-bb3a-4e2c-8821-9631ba4a569c" />

#### Output3 - Demand Forecast Output
<img width="994" height="562" alt="Screenshot 2025-12-15 203819" src="https://github.com/user-attachments/assets/77990f0f-8c1d-47a6-be52-d04304674833" />

<img width="999" height="495" alt="Screenshot 2025-12-15 203900" src="https://github.com/user-attachments/assets/f34efd53-942b-43ce-ba6c-3b5782840012" />

<img width="1006" height="499" alt="Screenshot 2025-12-15 203908" src="https://github.com/user-attachments/assets/71ae8dfd-3579-4d89-875e-c6811745e4c3" />

<img width="996" height="495" alt="Screenshot 2025-12-15 203916" src="https://github.com/user-attachments/assets/1ff6cb5a-2174-4912-9854-d94f6d5cc166" />

## Performance Metrics
1. Product Classification Accuracy: High accuracy achieved using DenseNet121.
2. Forecast Reliability: Prophet effectively captures demand trends and seasonality.
3. Inventory Decision Accuracy: Accurate reorder alerts based on forecasted demand.
   
## Results and Impact
The proposed system significantly improves retail inventory management by automating product identification and demand forecasting. By combining computer vision and time-series analysis, the system reduces manual errors, minimizes stock shortages, and prevents overstocking. The project demonstrates how AI-driven decision support can enhance operational efficiency and scalability in modern retail systems.

This solution provides a strong foundation for future advancements in smart retail analytics and intelligent supply chain management.

## Articles published / References
1. G. Huang et al., “Densely Connected Convolutional Networks,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
2. S. Taylor and B. Letham, “Forecasting at Scale,” The American Statistician, vol. 72, no. 1, 2018.
3. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.
4. Kaggle, “Retail Product Image Dataset,” 2023.



