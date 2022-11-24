def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    cleaned_df = pd.read_csv("cleaned_house_prices.csv")  # Importing the cleaned data

    # Create input and target variables
    X = cleaned_df.drop("prices",
                        axis=1)
    y = cleaned_df['prices']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=42,
                                                        test_size=0.2)
    tree_model = DecisionTreeRegressor(random_state=42,
                                       min_samples_split=2,
                                       max_depth=10)
    tree_model.fit(X_train,y_train)

    print(f"Accuracy with Decision tree model is : {tree_model.score(X_test,y_test)}")


if __name__ == "__main__":
    main()