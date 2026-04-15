import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def print_data(df) :
    # To visualize the data
    for index, row in df.iterrows():
        print(row)
        print("", end="\n\n")


def histogram_attack_per_country(df) :
    # Compter le nombre d'attaques par pays
    attacks_per_country = df["Country"].value_counts()

    # Histogramme (bar chart en réalité)
    plt.figure(figsize=(12, 6))
    attacks_per_country.plot(kind="bar")

    plt.title("Number of Cyber Attacks per Country")
    plt.xlabel("Country")
    plt.ylabel("Number of Attacks")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.yscale("log", base=10)
    plt.savefig("histogram_attack_per_country.png")
    plt.show()


def histogram_attack_per_industry(df) :
    attacks_per_industry = df["Target Industry"].value_counts()

    plt.figure(figsize=(12, 6))
    attacks_per_industry.plot(kind="bar")

    plt.title("Number of Cyber Attacks per Industry")
    plt.xlabel("Industry")
    plt.ylabel("Number of Attacks")

    plt.xticks(rotation=45)
    plt.yscale("log", base=10)
    plt.tight_layout()
    plt.savefig("histogram_attack_per_industry.png")
    plt.show()
    

def histogram_financial_loss(df):
    # Somme des pertes par pays
    loss_per_country = df.groupby("Country")["Financial Loss (in Million $)"].sum()

    # Trier
    loss_per_country = loss_per_country.sort_values(ascending=False)

    # Graphique
    plt.figure(figsize=(12, 6))
    loss_per_country.plot(kind="bar")

    plt.title("Total Financial Loss per Country")
    plt.xlabel("Country")
    plt.ylabel("Financial Loss (in Million $)")

    plt.xticks(rotation=45)
    plt.yscale("log", base=10)  # échelle logarithmique
    plt.tight_layout()
    plt.savefig("histogram_financial_loss.png")
    plt.show()
    

def boxplots_analysis(df):

    plt.figure(figsize=(12, 5))

    # Boxplot 1 : Number of Affected Users
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df["Number of Affected Users"])
    plt.title("Number of Affected Users")

    
    # Boxplot 2 : Incident Resolution Time
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["Incident Resolution Time (in Hours)"])
    plt.title("Incident Resolution Time (in Hours)")
    plt.tight_layout()
    plt.savefig("boxplots.png")

    plt.show()