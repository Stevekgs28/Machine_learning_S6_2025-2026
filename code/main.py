from functions import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# To read the CSV with pandas
df = pd.read_csv("../dataset.csv")

# Global preprocessing + quality report
df_clean, quality_report = preprocess_dataset(
    df,
    keep_unknown_as_category=True
)
print_data_quality_report(quality_report)

# Optional feature removal for this EDA version
if "Defense Mechanism Used" in df_clean.columns:
    df_clean = df_clean.drop(columns=["Defense Mechanism Used"])

# print_data(df_clean)
    
# Compter le nombre d'attaques par pays
histogram_attack_per_country(df_clean)

# Compter le nombre d'attaques par industrie
histogram_attack_per_industry(df_clean)

histogram_financial_loss(df_clean)

# Generic numeric distributions
plot_numeric_distributions(df_clean)

correlation(df_clean)