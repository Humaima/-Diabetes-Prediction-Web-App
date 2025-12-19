from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd
from preprocess import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    X_train, X_test, y_train, y_test, cols = preprocess_data()
    
    # Initialize and train Logistic Regression model
    model = LogisticRegression(
        penalty='l2',           # L2 regularization
        C=1.0,                 # Inverse of regularization strength
        solver='liblinear',     # Good for small datasets
        max_iter=1000,         # Increased for convergence
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save model
    joblib.dump(model, 'diabetes_model.joblib')
    
    # Print feature importance
    if hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': cols[:-1],  # Exclude 'Outcome' column
            'Importance': model.coef_[0]
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance (Logistic Regression Coefficients)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return model

if __name__ == '__main__':
    train_and_evaluate()