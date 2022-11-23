def main():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Importing the dataset
    df = pd.read_excel('Assignment-1_dataset.xlsx')
    print(df.columns)

    # Cleaning and Preprocessing the dataset
    # Creating a Price Column
    df['prices'] = df['House size (sqft)'] * df['House price of unit area'] # Calculating total price of houses

    # Normalized distance
    distance = df['Number of convenience stores'].values.reshape(-1, 1)
    scaler = StandardScaler()
    distance = scaler.fit_transform(distance)
    df['Number of convenience stores'] = distance

    # Dropping columns
    cleaned_df = df.drop(['Transaction date', 'House price of unit area'],
                         axis=1)  # Dropping prices per unit area because it is leads to Multicollinearity
    cleaned_df.to_csv("cleaned_house_prices.csv")


if __name__ == '__main__':
    main()