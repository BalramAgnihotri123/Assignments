def main():
    # Importing relevant
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    # getting the cleaned data
    df = pd.read_csv("cleaned_house_prices.csv")

    # Create Input and Target variables
    X = df.drop("prices",
                axis=1)
    y = df['prices']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=42,
                                                        train_size=0.8)

    # creating a random forest model
    forest = RandomForestRegressor(min_samples_split=42,
                                   n_estimators=50,
                                   random_state=42)

    forest.fit(X, y)

    print(f'\nAccuracy with the random forest model is : {forest.score(X_test, y_test) * 100:.2f} %')


if __name__ == "__main__":
    main()