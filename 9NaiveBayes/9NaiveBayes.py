# ==========================================
# NAIVE BAYES - YES / NO WITH VISUALIZATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------
# STEP 1: CREATE RANDOM DATASET
# ------------------------------------------

data = {
    "Weather": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
                "Overcast", "Sunny", "Sunny", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Mild", "Cool", "Mild", "Mild"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No",
             "Yes", "No", "Yes", "Yes"]
}

df = pd.DataFrame(data)

print("\nDataset:\n")
print(df)

# ------------------------------------------
# STEP 2: PRIOR PROBABILITIES
# ------------------------------------------

total = len(df)
yes_count = len(df[df["Play"] == "Yes"])
no_count = len(df[df["Play"] == "No"])

P_yes = yes_count / total
P_no = no_count / total

print("\nPrior Probabilities:")
print("P(Yes) =", round(P_yes, 3))
print("P(No)  =", round(P_no, 3))

# ------------------------------------------
# STEP 3: NEW SAMPLE
# ------------------------------------------

new_sample = {
    "Weather": "Sunny",
    "Temperature": "Cool"
}

print("\nNew Sample:", new_sample)

# ------------------------------------------
# STEP 4: LIKELIHOOD FUNCTION
# ------------------------------------------

def likelihood(feature, value, target):
    subset = df[df["Play"] == target]
    count = len(subset[subset[feature] == value])
    return count / len(subset)

# ------------------------------------------
# STEP 5: POSTERIOR CALCULATION
# ------------------------------------------

score_yes = P_yes
score_no = P_no

for feature in new_sample:
    score_yes *= likelihood(feature, new_sample[feature], "Yes")
    score_no *= likelihood(feature, new_sample[feature], "No")

print("\nPosterior Scores:")
print("Score Yes =", round(score_yes, 4))
print("Score No  =", round(score_no, 4))

# ------------------------------------------
# STEP 6: FINAL PREDICTION
# ------------------------------------------

prediction = "Yes" if score_yes > score_no else "No"

print("\nFinal Prediction:", prediction)

# ------------------------------------------
# STEP 7: VISUAL REPRESENTATION
# ------------------------------------------

labels = ["Yes", "No"]
scores = [score_yes, score_no]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, scores)

plt.title("Naive Bayes Posterior Probability Comparison")
plt.ylabel("Posterior Probability")

# Highlight predicted class
for i, bar in enumerate(bars):
    if labels[i] == prediction:
        bar.set_color("green")
    else:
        bar.set_color("red")

plt.tight_layout()

# Save image for GitHub
plt.savefig("NaiveBayes_YesNo_Output.png", dpi=300)

plt.show()
