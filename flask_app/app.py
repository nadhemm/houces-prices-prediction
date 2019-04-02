#!/usr/bin/env python
# coding: utf-8


from flask import Flask, request, render_template

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

app = Flask(__name__)


def fix_price(x):
    if isinstance(x, int) or isinstance(x, float): return x
    if x.split()[-1] == "DT": return int(x[:-3].replace(" ", ""))
    if x.split()[-1] == "Nuit": return 30 * int(x[:-10].replace(" ", ""))
    if x.split()[-1] == "Mois": return int(x[:-10].replace(" ", ""))
    if x.split()[-1] == "Semaine": return 4 * int(x[:-13].replace(" ", ""))


def fix_number(x):
    if str(x) == "nan": return x
    if str(x)[-1] == "+":
        x = int(str(x).strip()[:-1])
    if int(x) > 10:
        x = None
        return x
    return int(x)


def cast_type(x):
    if x == "vente": return 0
    return 1


def prepare_models():
    le = preprocessing.LabelEncoder()
    df = pd.read_csv("housing_prices_grand_tunis.csv")
    df.dropna(axis=0, subset=['price', 'address'], inplace=True)

    df = df[df.type == 'vente']

    # remove DT, /Mois, /Nuit...
    df['price'] = df['price'].map(fix_price)
    # remove '+' signs
    df['bathrooms'] = df['bathrooms'].map(fix_number)
    df['total_rooms'] = df['total_rooms'].map(fix_number)
    df['rooms'] = df['rooms'].map(fix_number)
    df['gov'] = df['gov'].apply(lambda x: x.strip())
    # replace vente/location by 0/1
    df['type'] = df['type'].map(cast_type)
    # if one of the 2 fields is empty try to fill it with the other one
    df['rooms'].fillna(df['total_rooms'], inplace=True)
    df['total_rooms'].fillna(df['rooms'], inplace=True)
    df['living_area'].fillna(df['total_area'], inplace=True)
    df['total_area'].fillna(df['living_area'], inplace=True)
    # remove row containing any non-appartment-related word
    df = df[df['title'].apply(lambda t: True if all(
        word not in t.split() for word in ["Terrain", "terrain", "أرض", "ارض", "lot", "lots"]) else False)]
    # label encoders for gov and address
    df[['gov', 'address']] = df[['gov', 'address']].apply(lambda x: le.fit_transform(x))
    df = pd.get_dummies(df, columns=['gov', 'address'])

    df.loc[df['type'] == 0, 'price'] = df.loc[df['type'] == 0, 'price'].apply(lambda x: 1000 * x if x < 1000 else x)

    df['total_rooms'].fillna(df['total_rooms'].median(), inplace=True)
    df['rooms'].fillna(df['rooms'].median(), inplace=True)
    df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
    df['living_area'].fillna(df['living_area'].median(), inplace=True)
    df['total_area'].fillna(df['total_area'].median(), inplace=True)

    # ## Linear Regression

    x = df[list(filter(lambda x: x not in ['price', 'title', 'type'], df.columns))]  # predictor
    y = df.price  # response
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # # Decision Tree

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=5)
    clf.fit(x_train, y_train)

    # # Random forest

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000)
    # Train the model on training data
    rf.fit(x_train, y_train)
    return linreg, clf, rf


def predict(model, total_surface, living_surface, number_of_rooms, total_rooms, bathrooms, gov, address):
    ADDRESS_LIST = ['address_Agba', 'address_Ariana Ville', 'address_Aïn Zaghouan', 'address_Borj Cedria', 'address_Borj El Amri', 'address_Borj Louzir', 'address_Boumhel', 'address_Carthage', 'address_Centre Urbain Nord', 'address_Centre Ville - Lafayette', 'address_Chotrana', 'address_Cité El Khadra', 'address_Cité Olympique', 'address_Denden', 'address_Djebel Jelloud', 'address_Djedeida', 'address_Douar Hicher', 'address_El Battan', 'address_El Kabaria', 'address_El Mourouj', 'address_El Omrane', 'address_El Omrane supérieur', 'address_El Ouardia', 'address_Ennasr', 'address_Ettadhamen', 'address_Ettahrir', 'address_Ezzahra', 'address_Ezzouhour', 'address_Fouchana', 'address_Gammarth', 'address_Ghazela', 'address_Hammam Chott', 'address_Hammam Lif', 'address_Hraïria', 'address_Jardin De Carthage', 'address_Jardins D''el Menzah', 'address_Kalaât El Andalous', 'address_L''aouina', 'address_La Goulette', 'address_La Marsa', 'address_La Soukra', 'address_Le Bardo', 'address_Le Kram', 'address_Les Berges Du Lac', 'address_Manar', 'address_Manouba Ville', 'address_Medina Jedida', 'address_Menzah', 'address_Mnihla', 'address_Mohamedia', 'address_Mornag', 'address_Mornaguia', 'address_Mutuelleville', 'address_Médina', 'address_Mégrine', 'address_Oued Ellil', 'address_Radès', 'address_Raoued', 'address_Sidi Daoud', 'address_Sidi El Béchir', 'address_Sidi Hassine', 'address_Sidi Thabet', 'address_Séjoumi', 'address_Tebourba']
    GOV_LIST = ['gov_Ariana', 'gov_Ben Arous', 'gov_Manouba', 'gov_Tunis']
    x = [int(bathrooms), int(total_rooms), int(living_surface), int(total_surface), int(number_of_rooms)]
    for el in GOV_LIST:
        if 'gov_' + gov == el: x.append(1)
        else: x.append(0)
    for el in ADDRESS_LIST:
        if 'address_' + address == el: x.append(1)
        else: x.append(0)
    print(x)
    return model.predict([x])[0]


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        model = request.form.get("model")
        total_surface = request.form.get("total_surface")
        living_surface = request.form.get("living_surface")
        number_of_rooms = request.form.get("number_of_rooms")
        total_rooms = request.form.get("total_rooms")
        bathrooms = request.form.get("bathrooms")
        gov = request.form.get("gov")
        address = request.form.get("address")
        print(str(request.form))
        if model == "linear_regression":
            model = lin_reg
        elif model == "decision_tree":
            model = dt
        elif model == "random_forest":
            model = rf
        price = predict(model=model, total_surface=total_surface, living_surface=living_surface,
                        number_of_rooms=number_of_rooms, total_rooms=total_rooms,
                        bathrooms=bathrooms, gov=gov, address=address)
        return render_template('home.html', price=price)
    return render_template('home.html')


if __name__ == '__main__':
    lin_reg, dt, rf = prepare_models()
    app.run()
