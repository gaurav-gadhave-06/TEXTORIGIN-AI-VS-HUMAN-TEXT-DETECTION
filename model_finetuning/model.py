from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load saved model and tokenizer
model_path = "model"  # or "./model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()  # Set model to evaluation mode

def predict(text):
    """
    Predict the class of the input text using the fine-tuned model.
    Returns class label, confidence, and a simple reason.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    # Map class to label
    label_map = {0: "Human", 1: "AI"}
    label = label_map[predicted_class]

    # Simple reason based on confidence
    if confidence > 0.8:
        reason = f"Model is highly confident this text is {label} generated."
    elif confidence > 0.6:
        reason = f"Model is somewhat confident this text is {label} generated."
    else:
        reason = f"Model is unsure, but predicts {label} generated."

    return label, confidence, reason

example_text = """
  
The Facial Action Coding system is valuable for students. One reason is that we can see how diffrent people felt in diffrent important events. The second is that the computer can change up the learning method if they see that the student is getting bored with the way they are currently learning. The last reason is that it could keep students more happy throughout the day. This is a huge improvement to the school system.

The first reason that the Facial Code System would be valuable is that we can see how difffent people felt during important events. For example they showed us the Mona Lisa and the results from it. If we look at all the imprtant events and see pictures of these people then we can see how the people really felt during these events. For example the Mona Lisa. In the painting she is smiling but it was one of the first paintings of people smiling so maybe she could have been worried or sad, but the results stated for the most part she was actully happy.

The second reason is that the computer in a classroom coud see how we feel and if we become bored and uninterested. In paragraph 6 it states , " A classroom comupter could recogniz when a student is becominf confused or bored. Then it could modify the lesson, like an effective instructor." So overall we would learn better in the sense that it would switch up the lesson so that we woudl stay focused. It would keep us interested. Also overall we would learn the material faster since we wouldn't be procrastinating as much.

The final reason is that it could keep students more happy throiughout the day, In paragraph 9 in states, " According to the Facial Feedback Theory of Emorion. moving your facial muscles not only expresses emotions , but also may even help produce them." So if a student is not feeling well or is sad they won't learn as well. So the computer will sense then and give them a break or an activity that they like to do and their mood will increase. They will learn better because of it.

The Facial Action Coding system is valuable for students. The first reason reason is that we can see how diffrent people felt in diffrent important events. Secondly is that the computer can change up the learning method if they see that the student is getting bored with the way they are currently learning. The last reason is that it could keep students more happy throughout the day. This could be a huge improvement to the school system.

"""
label, confidence, reason = predict(example_text)
print(f"Predicted class for the input text: {label}")
print(f"Confidence score: {confidence:.2f}")
print(f"Reason: {reason}")