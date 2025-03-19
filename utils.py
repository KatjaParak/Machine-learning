import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_file():
    cardio = pd.read_csv("cardio_train.csv", sep=';')
    return cardio


def plot_presence_absence(cardio):
    fig = plt.pie(cardio['cardio'].value_counts(), autopct='%1.2f%%', labels=['presence', 'absence'],
                  colors=['darkslategray', 'powderblue'], explode=[0.02, 0.02], startangle=90)
    plt.title(
        "Presence or absence of cardiovascular disease in patients", fontsize=10)
    return fig


def plot_cholesterol_levels(cardio):
    fig = plt.pie(cardio['cholesterol'].value_counts(), labels=['normal', 'above normal', 'well above normal'], explode=[0.02, 0.02, 0.02],
                  autopct='%1.1f%%', colors=['darkslategrey', 'cadetblue', 'powderblue'])
    plt.title("Cholesterol levels in patients", fontsize=10, loc='center')
    return fig


def plot_age_distribution(cardio):
    fig = sns.histplot(data=[i/365 for i in cardio['age']],
                       bins=100, color='powderblue')
    plt.title("Age distribution of patients", fontsize=10, loc='center')
    plt.xlabel("Age(years)")
    return fig


def plot_proportion_smokers(cardio):
    fig = plt.pie(cardio['smoke'].value_counts(), labels=['Non-smokers', 'smokers'], explode=[0.02, 0.02], autopct='%1.1f%%',
                  colors=['darkslategray', 'powderblue'])
    plt.title("Proportion of smokers in patients", fontsize=10, loc='center')
    return fig


def plot_weight_distribution(cardio):
    fig = sns.histplot(cardio['weight'], bins=100, color='powderblue')
    plt.title("Weight distribution of patients", fontsize=10, loc='center')
    return fig


def plot_height_distribution(cardio):
    fig = sns.displot(cardio['height'], bins=100, color='cadetblue')
    plt.title("Height distribution of patients", fontsize=10, loc='center')
    plt.xlabel("Height(cm)")
    return fig


def plot_gender_distribution(cardio):
    fig = plt.pie(cardio[cardio['cardio'] == 1]['gender'].value_counts(), autopct='%1.2f%%', labels=['women', 'men'],
                  colors=['darkslategray', 'powderblue'], explode=[0.02, 0.02], startangle=90)
    plt.title("Gender distribution of positively diagnosed patients", fontsize=10)
    return fig
