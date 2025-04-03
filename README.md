# Credit Card Fraud Detection with Graph Neural Networks (GNNs)

## Overview
This project aims to detect fraudulent credit card transactions using *Graph Neural Networks (GNNs)*. Traditional fraud detection techniques focus on individual transactions, missing the complex relationships between users, merchants, and purchases. Our system leverages graph-based models to uncover hidden fraud patterns.

## Features
- *Graph-based fraud detection* using GNNs
- *Real-world dataset* from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- *Transaction relationship modeling* for better fraud detection
- *Web-based interface* for fraud visualization

## Project Structure
```
credit-card-detect/
├── app/
│   ├── static/
│   │   └── css/
│   │       └── styles.css   # Frontend styling
│   ├── templates/
│   │   └── index.html       # Web application interface
│   ├── __init__.py
│   └── routes.py            # Backend routes
├── models/
│   ├── __init__.py
│   └── gnn_model.py         # Graph Neural Network model
├── data/
│   └── creditcard.csv       # Credit card transaction dataset
├── run.py                   # Script to run the application
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Steps
1. *Clone the Repository:*
   sh
   git clone https://github.com/Ro706/credit_card_fraud_detection_app.git
   cd credit_card_fraud_detection_app
   

2. *Create a Virtual Environment & Activate it:*
   sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   

3. *Install Dependencies:*
   sh
   pip install -r requirements.txt
   

4. *Run the Application:*
   sh
   python run.py
   

## Dataset
We used the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized transaction data. Each transaction includes:
- *Number of times a user pays* a certain amount
- *Transaction amount* in dollars
- Anonymized transaction features
- Labels indicating *fraudulent or legitimate* transactions

## Model Implementation
- *Graph Construction:* Transactions, users, and merchants are represented as *nodes, while interactions form **edges*.
- *Graph Neural Network (GNN):* Uses *Graph Convolutional Networks (GCN)* to analyze connections and detect anomalies.
- *Training & Evaluation:* The model learns from historical fraud cases, predicting fraudulent transactions with high accuracy.

## Results
- The GNN model *outperforms traditional fraud detection methods*.
- *Graph-based relationships* help identify new fraud patterns that rule-based models miss.

## Future Work
- *Real-time fraud detection* for immediate alerts
- *Improved scalability* for large transaction networks
- *Self-learning GNN models* to adapt to evolving fraud tactics

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.
