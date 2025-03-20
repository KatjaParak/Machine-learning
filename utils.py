import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_file():
    cardio = pd.read_csv("cardio_train.csv", sep=';')
    return cardio


def calculate_bmi(cardio):
    bmi = np.round(np.divide(cardio['weight'],
                   np.power(cardio['height']*0.01, 2)))
    return bmi


def remove_outliers(cardio, feature):
    q3 = np.quantile(cardio[feature], 0.75)
    q1 = np.quantile(cardio[feature], 0.25)
    IQR = q3 - q1
    return cardio[(cardio[feature] <= (q3 + 3 * IQR)) & (cardio[feature] >= (q1 - 3 * IQR))]


def create_bmi_class(cardio):
    return pd.cut(x=cardio['bmi'], bins=[18.5, 25, 30, 35, 40, 50],
                  labels=['normal range', 'overweight', 'obese(class I)', 'obese (class II)', 'obese (class III)'])


# the code was sourced from statology.org
def set_pressure_category(cardio):
    conditions = [
        (cardio['ap_hi'] <= 120) & (cardio['ap_lo'] <= 80),
        ((cardio['ap_hi'] >= 120) & (cardio['ap_hi'] <= 129)) & (
            cardio['ap_lo'] <= 80),
        ((cardio['ap_hi'] >= 130) & (cardio['ap_hi'] <= 139)) & (
            (cardio['ap_lo'] <= 90) & (cardio['ap_lo'] >= 80)),
        ((cardio['ap_hi'] >= 140) & (cardio['ap_hi'] <= 180)) & (
            cardio['ap_lo'] >= 90),
        ((cardio['ap_hi'] >= 180) & (cardio['ap_hi'] <= 200)) & (
            cardio['ap_lo'] >= 120)
    ]

    category = ['healthy', 'elevated', 'stage_1_hyper',
                'stage_2_hyper', 'hyper_crisis']
    cardio['pressure_category'] = np.select(conditions, category)

    filtered_cardio = cardio[cardio['pressure_category'] != '0']
    return filtered_cardio


def plot_corr_matrix(filtered_cardio):
    filtered_cardio = pd.get_dummies(filtered_cardio).corr()
    plt.figure(figsize=(10, 8))
    fig = sns.heatmap(filtered_cardio, linewidths=.2, annot=True, annot_kws={
                      'size': 6}, fmt='.1f', cmap='crest')
    plt.title("Correlation coefficients")
    return fig


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
