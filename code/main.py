from functions import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# To read the CSV with pandas
df = pd.read_csv("../dataset.csv")

# To clean the dataset
print(df.isnull().sum())
df = df.dropna()

# To remove this column : we don't need it for our project
df = df.drop(columns=["Defense Mechanism Used"])

# print_data(df)
    
# Compter le nombre d'attaques par pays
histogram_attack_per_country(df)

# Compter le nombre d'attaques par industrie
histogram_attack_per_industry(df)

histogram_financial_loss(df)

boxplots_analysis(df)