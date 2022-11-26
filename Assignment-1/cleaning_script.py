import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # Importing the dataset
    df = pd.read_excel('Assignment-1_dataset.xlsx')
    print(df.columns)
    # Cleaning and Preprocessing the dataset
    # Creating a Price Column
    df['prices'] = df['House size (sqft)'] * df['House price of unit area']  # Calculating total price of houses

    # Getting rid of the outliers
    df= remove_outliers(dataframe=df,
                        column_name="prices")

    # feature engineering for location columns
    location = df[['latitude', 'longitude']]
    kmeans = KMeans(4)
    df['location'] = kmeans.fit_predict(location)

    # Normalized distance
    distance = df['Distance from nearest Metro station (km)'].values.reshape(-1, 1)
    scaler = StandardScaler()
    distance = scaler.fit_transform(distance)
    df["Distance from nearest Metro station (km)"] = distance

    # Normalization of columns
    df = scaling_columns(dataframe=df,
                         column_names=['Distance from nearest Metro station (km)'])

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


def scaling_columns(dataframe: pd.DataFrame,
                    column_names: list):
    scaler = StandardScaler()

    scaled_column = dataframe[column_names].values
    scaled_column = scaler.fit_transform(scaled_column.reshape(len(dataframe), len(column_names)))
    dataframe[column_names] = scaled_column

    return dataframe


if __name__ == '__main__':
    main()
