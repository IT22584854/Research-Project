import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv("outputs/evaluation_results.csv")

os.makedirs("outputs/plots", exist_ok=True)

# histogram of Final Scores
plt.figure()
plt.hist(df["final_score"], bins=10)
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.title("Distribution of Evaluation Scores")
plt.savefig("outputs/plots/final_score_distribution.png")
plt.close()

# boxplot by Profile
plt.figure()
df.boxplot(column="final_score", by="profile")
plt.xlabel("Profile")
plt.ylabel("Final Score")
plt.title("Score Distribution by Evaluation Profile")
plt.suptitle("")
plt.savefig("outputs/plots/score_by_profile.png")
plt.close()

# boxplot by Category
plt.figure()
df.boxplot(column="final_score", by="category", rot=45)
plt.xlabel("Category")
plt.ylabel("Final Score")
plt.title("Score Distribution by Question Category")
plt.suptitle("")
plt.tight_layout()
plt.savefig("outputs/plots/score_by_category.png")
plt.close()

print("Plots saved in outputs/plots/")
