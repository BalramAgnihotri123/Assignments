def main():
    # Importing relevant Libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    cleaned_df = pd.read_csv("cleaned_house_prices.csv")  # Importing the cleaned data

    # Create input and target variables
    X = cleaned_df.drop("prices",
                        axis=1)
    y = cleaned_df['prices']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=42,
                                                        test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"accuracy with Linear model is : {model.score(X_test, y_test)}")


if __name__ == '__main__':
    main()
