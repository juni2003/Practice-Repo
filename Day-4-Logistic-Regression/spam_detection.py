import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from logistic_regression_from_scratch import LogisticRegressionScratch
from classification_metrics import confusion_matrix, print_all_metrics, plot_confusion_matrix

spam_emails = [
    "Congratulations! You've won a free lottery prize. Click here to claim.",
    "Get rich quick! Make money from home with this one simple trick.",
    "URGENT: Your account will be closed. Verify your information now.",
    "Free iPhone! Limited time offer. Act now before it's too late.",
    "You have been selected for a special discount. Buy now!",
    "Cheap medications available. No prescription needed.",
    "Make $5000 per week working from home. No experience required.",
    "Your credit card has been charged. Click here if this wasn't you.",
    "Hot singles in your area want to meet you!",
    "Lose 20 pounds in 2 weeks with this miracle pill.",
    "Claim your free gift card now! Only 5 remaining.",
    "Investment opportunity: Double your money in 30 days guaranteed.",
    "Your package could not be delivered. Update shipping info.",
    "Congratulations! You qualify for a loan up to $50000.",
    "Get free access to premium content. Click here now.",
]

legitimate_emails = [
    "Meeting scheduled for tomorrow at 3pm in conference room B.",
    "Please review the attached document and provide your feedback.",
    "Your monthly statement is now available for viewing.",
    "Thank you for your recent purchase. Your order has shipped.",
    "Reminder: Team standup meeting at 10am today.",
    "Could you please send me the project report by Friday?",
    "Your subscription renewal is coming up next month.",
    "We appreciate your business. Contact us if you need assistance.",
    "Your appointment is confirmed for next Tuesday at 2pm.",
    "The quarterly results have been published on the company portal.",
    "Please complete the mandatory training module by end of week.",
    "Your password was successfully changed on your account.",
    "Welcome to our service. Here is your getting started guide.",
    "Invoice for your recent order is attached to this email.",
    "Project deadline has been extended to next Friday.",
]

def prepare_email_dataset():
    """
    Prepare the email dataset for classification.
    
    Steps:
    1. Combine spam and legitimate emails
    2. Create labels (1 for spam, 0 for legitimate)
    3. Convert text to numerical features using Bag of Words
    
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Labels
    vectorizer : CountVectorizer
        Fitted vectorizer for future use
    """
    emails = spam_emails + legitimate_emails
    
    labels = np.array([1] * len(spam_emails) + [0] * len(legitimate_emails))
    
    vectorizer = CountVectorizer(
        max_features=50,
        stop_words='english',
        lowercase=True
    )
    
    X = vectorizer.fit_transform(emails).toarray()
    
    return X, labels, vectorizer


def analyze_features(vectorizer):
    """
    Analyze the most important words (features) in the dataset.
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\nTotal features extracted: {len(feature_names)}")
    print("\nTop 20 features (words):")
    print("-"*60)
    for i, feature in enumerate(feature_names[:20], 1):
        print(f"{i:2d}. {feature}")
    print("="*60)


def predict_new_emails(model, scaler, vectorizer):
    """
    Test the trained model on new, unseen emails.
    """
    print("\n" + "="*60)
    print("TESTING ON NEW EMAILS")
    print("="*60 + "\n")
    
    new_emails = [
        "Click here to get your free prize money now!",
        "Team meeting rescheduled to 4pm tomorrow",
        "You won the lottery! Claim your prize immediately.",
        "Please find attached the quarterly sales report",
        "Lose weight fast with our new miracle supplement",
        "Your order has been shipped and will arrive Tuesday"
    ]
    
    expected_labels = [1, 0, 1, 0, 1, 0]
    
    X_new = vectorizer.transform(new_emails).toarray()
    X_new_scaled = scaler.transform(X_new)
    
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)
    
    print(f"{'Email':<50} {'Prediction':<15} {'Probability':<15} {'Correct?'}")
    print("-"*95)
    
    for i, email in enumerate(new_emails):
        pred_label = "SPAM" if predictions[i] == 1 else "LEGITIMATE"
        prob = probabilities[i]
        is_correct = "✓" if predictions[i] == expected_labels[i] else "✗"
        
        short_email = email[:47] + "..." if len(email) > 50 else email
        print(f"{short_email:<50} {pred_label:<15} {prob:<15.4f} {is_correct}")
    
    accuracy = np.mean(predictions == np.array(expected_labels))
    print("-"*95)
    print(f"Accuracy on new emails: {accuracy:.2f} ({accuracy*100:.0f}%)")
    print("="*60)


def plot_spam_vs_legitimate_distribution(X, y, feature_names):
    """
    Visualize the distribution of key spam-related words.
    """
    spam_words = ['free', 'click', 'win', 'money', 'offer']
    
    spam_indices = np.where(y == 1)[0]
    legit_indices = np.where(y == 0)[0]
    
    available_words = []
    spam_counts = []
    legit_counts = []
    
    for word in spam_words:
        if word in feature_names:
            word_idx = np.where(feature_names == word)[0][0]
            available_words.append(word)
            spam_counts.append(np.sum(X[spam_indices, word_idx]))
            legit_counts.append(np.sum(X[legit_indices, word_idx]))
    
    if not available_words:
        print("Note: Spam-related words not found in features")
        return
    
    x = np.arange(len(available_words))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, spam_counts, width, label='Spam Emails', color='#e74c3c', edgecolor='black')
    plt.bar(x + width/2, legit_counts, width, label='Legitimate Emails', color='#2ecc71', edgecolor='black')
    
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Word Frequency: Spam vs Legitimate Emails', fontsize=14, fontweight='bold')
    plt.xticks(x, available_words)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('spam_word_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Word distribution plot saved as 'spam_word_distribution.png'")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMAIL SPAM DETECTION USING LOGISTIC REGRESSION")
    print("="*60 + "\n")
    
    print("Step 1: Preparing email dataset...")
    X, y, vectorizer = prepare_email_dataset()
    print(f"Total emails: {len(y)}")
    print(f"Spam emails: {np.sum(y
