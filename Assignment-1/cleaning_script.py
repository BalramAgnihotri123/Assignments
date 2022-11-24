import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # Importing the dataset
    df = pd.read_excel('Assignment-1_dataset.xlsx')
    print(df.columns)
    # Cleaning and Preprocessing the dataset
    # Creating a Price Column
    df['prices'] = df['House size (sqft)'] * df['House price of unit area']  # Calculating total price of houses

    # Getting rid of the outliers
    df = remove_outliers(dataframe=df,
                         column_name="prices")

    # Normalized distance
    distance = df['Distance from nearest Metro station (km)'].values.reshape(-1, 1)
    scaler = StandardScaler()
    distance = scaler.fit_transform(distance)
    df['Distance from nearest Metro station (km)'] = distance

    # Dropping columns
    cleaned_df = df.drop(['Transaction date', 'House price of unit area'],
                         axis=1)  # Dropping prices per unit area because it is leads to Multicollinearity
    cleaned_df.to_csv("cleaned_house_prices.csv")


def remove_outliers(dataframe: pd.DataFrame,
                    column_name: str) -> object:
    """
    inputs in a dataframe df and a column to consider for outlier removal
    returns a cleaned df with removed outliers
    """
    q1 = dataframe[column_name].quantile(0.25)
    q3 = dataframe[column_name].quantile(0.75)
    iqr = q3 - q1
    upper = np.where(dataframe[column_name] >= (q3 + 1.5 * iqr))
    dataframe.drop(list(upper[0]),
                   inplace=True)
    dataframe = dataframe.reset_index().drop("index",
                                             axis=1)
    return dataframe


if __name__ == '__main__':
    main()
